use crate::{cmd::CliArgs, FrameReceiver, MatchedFrameSender, FRAMES_PROCESSED};
use color_eyre::eyre::{eyre, ContextCompat, Result, WrapErr};
use opencv::{
	core::{self, GpuMat, Point, Ptr, Size, Stream, CV_8U},
	cudaarithm,
	cudaimgproc::{self, CUDA_TemplateMatching},
	imgproc::TM_CCOEFF_NORMED,
	prelude::{CUDA_TemplateMatchingTrait, GpuMatTrait, Mat},
};
use std::{
	borrow::Cow,
	sync::{atomic::Ordering, Arc},
	thread,
};

pub fn process_frame(
	args: &CliArgs,
	stream: &mut Stream,
	result: &mut GpuMat,
	matching: &mut Ptr<CUDA_TemplateMatching>,
	frame: &GpuMat,
	template: &GpuMat,
) -> Result<bool> {
	let frame = match args.bounds {
		Some(bounds) => Cow::Owned(GpuMat::roi(frame, bounds).wrap_err("invalid roi")?),
		None => Cow::Borrowed(frame),
	};

	matching
		.match_(&*frame, template, result, stream)
		.wrap_err("failed to match template")?;

	let mut _min_val = 0.0;
	let mut max_val = 0.0;
	let mut _min_loc = Point::default();
	let mut _max_loc = Point::default();
	cudaarithm::min_max_loc(
		result,
		&mut _min_val,
		&mut max_val,
		&mut _min_loc,
		&mut _max_loc,
		&core::no_array(),
	)
	.wrap_err("calculating global maximum value failed")?;

	Ok(max_val >= args.threshold)
}

pub fn worker_thread(
	args: &CliArgs,
	template: Mat,
	frame_receiver: FrameReceiver,
	result_sender: MatchedFrameSender,
) -> Result<()> {
	let mut template_gpu = GpuMat::default().wrap_err("failed to create template gpu mat")?;
	template_gpu
		.upload(&template)
		.wrap_err("failed to upload template to gpu")?;
	let mut result = GpuMat::default().wrap_err("failed to create result gpu mat")?;
	let mut matching =
		cudaimgproc::create_template_matching(CV_8U, TM_CCOEFF_NORMED, Size::default())
			.wrap_err("failed to setup cuda template matching")?;
	let mut stream = Stream::default().wrap_err("failed to create stream")?;
	let mut frame_gpu = GpuMat::default().wrap_err("failed to create frame gpu mat")?;
	for (frame, frame_num) in frame_receiver.iter() {
		frame_gpu
			.upload(&frame)
			.wrap_err("failed to upload frame to gpu")?;
		if process_frame(
			args,
			&mut stream,
			&mut result,
			&mut matching,
			&frame_gpu,
			&template_gpu,
		)
		.wrap_err_with(|| format!("failed to process frame {frame_num} on gpu"))?
		{
			result_sender.send(frame_num).map_err(|_| {
				eyre!("failed to send result for frame {frame_num} back to main thread")
			})?;
		}
		FRAMES_PROCESSED.fetch_add(1, Ordering::Relaxed);
	}
	Ok(())
}

pub fn spawn_threads(
	args: Arc<CliArgs>,
	template: &Mat,
	frame_receiver: FrameReceiver,
	result_sender: MatchedFrameSender,
) -> Result<()> {
	let core_ids = core_affinity::get_core_ids().wrap_err("failed to get CPU core IDs")?;
	for id in core_ids {
		let args = args.clone();
		let template = template.clone();
		let frame_receiver = frame_receiver.clone();
		let result_sender = result_sender.clone();
		thread::Builder::new()
			.name(format!("gpu worker core {}", id.id))
			.spawn(move || {
				if !core_affinity::set_for_current(id) {
					eprintln!("failed to set thread affinity for core {}", id.id);
				}
				worker_thread(&args, template, frame_receiver, result_sender)
					.expect("gpu worker thread errored");
			})
			.wrap_err_with(|| format!("failed to spawn gpu worker on core {}", id.id))?;
	}
	Ok(())
}
