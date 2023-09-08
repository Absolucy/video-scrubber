use crate::cmd::ScrubArgs;
use color_eyre::eyre::{ContextCompat, Result, WrapErr};
use crossbeam_channel::unbounded;
use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use parking_lot::Mutex;
use std::{
	fmt::Write,
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc,
	},
	thread,
};
use thread_priority::{ThreadBuilderExt, ThreadPriority};
use video_scrubber_core::{
	frame::{self, Frame},
	opencv::{
		imgcodecs::IMREAD_GRAYSCALE,
		videoio::{VideoCapture, VideoCaptureTraitConst, CAP_FFMPEG, CAP_PROP_FRAME_COUNT},
	},
	segments, templates, video, FRAMES_PROCESSED,
};

pub static DONE_PROCESSING: AtomicBool = AtomicBool::new(false);

pub fn scrub(args: ScrubArgs) -> Result<()> {
	let (frame_sender, frame_receiver) = unbounded::<Frame>();
	let (result_sender, result_receiver) = unbounded::<usize>();
	let exceeding_frames = Arc::new(Mutex::new(Vec::<usize>::new()));

	if let Some(opts) = &args.ffmpeg_opts {
		std::env::set_var("OPENCV_FFMPEG_CAPTURE_OPTIONS", opts);
	}

	// Read in the video file
	let mut capture = VideoCapture::from_file(
		args.input
			.to_str()
			.wrap_err("invalid input path cannot be represented as a str")?,
		CAP_FFMPEG,
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

	let pos_templates = templates::load_multi(&args.pos_templates, IMREAD_GRAYSCALE)
		.wrap_err("failed to parse positive templates")?;
	let neg_templates = templates::load_multi(&args.neg_templates, IMREAD_GRAYSCALE)
		.wrap_err("failed to parse negative templates")?;

	frame::cpu::spawn_threads(
		args.threads,
		args.bounds,
		&pos_templates,
		&neg_templates,
		Some(args.pos_threshold),
		Some(args.neg_threshold),
		frame_receiver.clone(),
		result_sender.clone(),
	)
	.wrap_err("failed to setup cpu worker threads")?;

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

	let segments = segments::frames_to_segments(args.padding, &capture, &*exceeding_frames)
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
	video::splice_video(&args.input, &args.output, &segments)
		.wrap_err("failed to splice segments into single video")?;
	println!("finished splicing video");

	Ok(())
}
