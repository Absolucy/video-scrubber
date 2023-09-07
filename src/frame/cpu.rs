use crate::{cmd::CliArgs, FrameReceiver, MatchedFrameSender, FRAMES_PROCESSED};
use color_eyre::eyre::{eyre, ContextCompat, Result, WrapErr};
use opencv::{
	core::{self, Mat},
	imgproc,
};
use std::{
	sync::{atomic::Ordering, Arc},
	thread,
};

pub fn process_frame(args: &CliArgs, result: &mut Mat, frame: Mat, template: &Mat) -> Result<bool> {
	let frame = match args.bounds {
		Some(bounds) => Mat::roi(&frame, bounds).wrap_err("invalid roi")?,
		None => frame,
	};

	// Perform template matching on the ROI
	imgproc::match_template(
		&frame,
		template,
		result,
		imgproc::TM_CCOEFF_NORMED,
		&core::no_array(),
	)
	.wrap_err("template matching failed")?;

	// Check for a match with a threshold
	let mut max_val: f64 = 0.0;
	core::min_max_loc(
		result,
		None,
		Some(&mut max_val),
		None,
		None,
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
	let mut result = Mat::default();
	for (frame, frame_num) in frame_receiver.iter() {
		if process_frame(args, &mut result, frame, &template)
			.wrap_err_with(|| format!("failed to process frame {frame_num} on cpu"))?
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
	for id in core_ids
		.iter()
		.copied()
		.take(args.threads.unwrap_or(usize::MAX))
	{
		let args = args.clone();
		let template = template.clone();
		let frame_receiver = frame_receiver.clone();
		let result_sender = result_sender.clone();
		thread::Builder::new()
			.name(format!("cpu worker core {}", id.id))
			.spawn(move || {
				if !core_affinity::set_for_current(id) {
					eprintln!("failed to set thread affinity for core {}", id.id);
				}
				worker_thread(&args, template, frame_receiver, result_sender)
					.expect("cpu worker thread errored");
			})
			.wrap_err_with(|| format!("failed to spawn cpu worker on core {}", id.id))?;
	}
	Ok(())
}
