pub mod cmd;
pub mod frame;
pub mod segments;
pub mod video;

use clap::Parser;
use color_eyre::eyre::{ContextCompat, Result, WrapErr};
use crossbeam_channel::{unbounded, Receiver, Sender};
use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use opencv::{
	core::{get_cuda_enabled_device_count, Mat, Vector},
	imgcodecs::{self, IMREAD_COLOR},
	videoio::{
		VideoCapture, VideoCaptureProperties, VideoCaptureTraitConst, CAP_ANY, CAP_PROP_FRAME_COUNT,
	},
};
use parking_lot::Mutex;
use std::{
	fmt::Write,
	sync::{
		atomic::{AtomicBool, AtomicUsize, Ordering},
		Arc,
	},
	thread,
};
use thread_priority::{ThreadBuilderExt, ThreadPriority};

pub type Frame = (Mat, usize);
pub type FrameSender = Sender<Frame>;
pub type FrameReceiver = Receiver<Frame>;
pub type MatchedFrameSender = Sender<usize>;
pub type MatchedFrameReceiver = Receiver<usize>;

#[global_allocator]
static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

pub static FRAMES_PROCESSED: AtomicUsize = AtomicUsize::new(0);
pub static DONE_PROCESSING: AtomicBool = AtomicBool::new(false);

fn main() -> Result<()> {
	color_eyre::install().wrap_err("failed to install color eyre handler")?;
	let args = Arc::new(cmd::CliArgs::parse());

	let (frame_sender, frame_receiver) = unbounded::<Frame>();
	let (result_sender, result_receiver) = unbounded::<usize>();
	let exceeding_frames = Arc::new(Mutex::new(Vec::<usize>::new()));

	let capture_properties =
		Vector::from_slice(&[VideoCaptureProperties::CAP_PROP_HW_ACCELERATION as i32, 1]);

	// Read in the video file
	let mut capture = VideoCapture::from_file_with_params(
		args.input
			.to_str()
			.wrap_err("invalid input path cannot be represented as a str")?,
		CAP_ANY,
		&capture_properties,
	)
	.wrap_err_with(|| format!("failed to read video from {}", args.input.display()))?;

	let total_frames = capture
		.get(CAP_PROP_FRAME_COUNT)
		.wrap_err("failed to get frame count property from video")?;

	let exceeding_frames_clone = exceeding_frames.clone();
	thread::spawn(move || {
		for frame in result_receiver {
			exceeding_frames_clone.lock().push(frame);
		}
	});

	// Spawn worker threads
	let template = imgcodecs::imread(
		args.template
			.to_str()
			.wrap_err("invalid template path cannot be represented as a str")?,
		IMREAD_COLOR,
	)
	.wrap_err_with(|| format!("failed to read image from {}", args.template.display()))?;

	if args.cuda {
		if get_cuda_enabled_device_count().unwrap_or(0) <= 0 {
			panic!("no CUDA devices found!");
		}
		#[cfg(feature = "cuda")]
		{
			println!("using CUDA");
			frame::gpu::spawn_threads(
				args.clone(),
				&template,
				frame_receiver.clone(),
				result_sender.clone(),
			)
			.wrap_err("failed to setup gpu worker threads")?;
		}
		#[cfg(not(feature = "cuda"))]
		{
			panic!("CUDA not supported!")
		}
	} else {
		println!("using CPU");
		frame::cpu::spawn_threads(
			args.clone(),
			&template,
			frame_receiver.clone(),
			result_sender.clone(),
		)
		.wrap_err("failed to setup cpu worker threads")?;
	}

	let progress_thread = thread::Builder::new()
		.name("frame progress thread".to_owned())
		.spawn_with_priority(ThreadPriority::Min, move |_| {
			let total_frames = total_frames.round() as u64;
			let progress_bar = ProgressBar::new(total_frames).with_style(
				ProgressStyle::with_template(
					"[{elapsed}] {wide_bar:.green/red} {pos}/{len} frames ({per_sec}, ETA: {eta})",
				)
				.unwrap()
				.with_key("pos", |state: &ProgressState, w: &mut dyn Write| {
					write!(w, "{}", HumanCount(state.pos())).unwrap()
				})
				.with_key("len", |state: &ProgressState, w: &mut dyn Write| {
					write!(w, "{}", HumanCount(state.len().unwrap())).unwrap()
				})
				.with_key("per_sec", |state: &ProgressState, w: &mut dyn Write| {
					write!(w, "{:.1} fps", state.per_sec().round() as u64).unwrap()
				}),
			);
			while !DONE_PROCESSING.load(Ordering::Relaxed) {
				let frames_processed = FRAMES_PROCESSED.load(Ordering::Relaxed) as u64;
				if frames_processed >= total_frames {
					DONE_PROCESSING.store(true, Ordering::Relaxed);
					break;
				}
				progress_bar.set_position(frames_processed);
				std::thread::yield_now();
			}
			progress_bar.finish();
		})
		.wrap_err("failed to spawn progress bar thread")?;

	frame::send_frames(&mut capture, frame_sender)
		.wrap_err("failed to send frames to worker threads")?;

	drop(frame_receiver);
	drop(result_sender);
	let _ = progress_thread.join();
	println!("finished scanning video");
	DONE_PROCESSING.store(true, Ordering::Relaxed);

	let mut exceeding_frames = exceeding_frames.lock();
	exceeding_frames.sort(); // Sort the frames in ascending order

	let segments = segments::frames_to_segments(&args, &capture, &exceeding_frames)
		.wrap_err("failed to convert frames to time ranges")?;

	for (idx, (start, end)) in segments.iter().copied().enumerate() {
		println!("segment #{idx}: {start:.1}s -> {end:.1}s");
	}

	let percentage_exceeding = (exceeding_frames.len() as f64 / total_frames) * 100.0;
	println!(
		"found {} (out of {}) exceeding frames ({:.2}%)",
		exceeding_frames.len(),
		total_frames,
		percentage_exceeding
	);

	println!("splicing video");
	video::splice_video(&args, &segments)
		.wrap_err("failed to splice segments into single video")?;
	println!("finished splicing video");

	Ok(())
}
