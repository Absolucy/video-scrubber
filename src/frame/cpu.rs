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

pub fn process_frame(
	args: &CliArgs,
	result: &mut Mat,
	frame: Mat,
	templates: &[Mat],
) -> Result<bool> {
	let frame = match args.bounds {
		Some(bounds) => Mat::roi(&frame, bounds).wrap_err("invalid roi")?,
		None => frame,
	};

	let mut max_val: f64 = 0.0;
	for template in templates {
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
		core::min_max_loc(
			result,
			None,
			Some(&mut max_val),
			None,
			None,
			&core::no_array(),
		)
		.wrap_err("calculating global maximum value failed")?;
		if max_val >= args.threshold {
			return Ok(true);
		}
	}

	Ok(false)
}

pub fn worker_thread(
	args: &CliArgs,
	templates: Vec<Mat>,
	frame_receiver: FrameReceiver,
	result_sender: MatchedFrameSender,
) -> Result<()> {
	let mut result = Mat::default();
	for (frame, frame_num) in frame_receiver.iter() {
		if process_frame(args, &mut result, frame, &templates)
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
	templates: &[Mat],
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
		let templates = templates.to_vec();
		let frame_receiver = frame_receiver.clone();
		let result_sender = result_sender.clone();
		thread::Builder::new()
			.name(format!("cpu worker core {}", id.id))
			.spawn(move || {
				if !core_affinity::set_for_current(id) {
					eprintln!("failed to set thread affinity for core {}", id.id);
				}
				worker_thread(&args, templates, frame_receiver, result_sender)
					.expect("cpu worker thread errored");
			})
			.wrap_err_with(|| format!("failed to spawn cpu worker on core {}", id.id))?;
	}
	Ok(())
}
