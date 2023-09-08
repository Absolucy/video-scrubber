#![allow(clippy::too_many_arguments)]
use crate::{
	frame::{Frame, FrameReceiver, MatchedFrameSender},
	FRAMES_PROCESSED,
};
use color_eyre::eyre::{eyre, ContextCompat, Result, WrapErr};
use opencv::{
	core::{self, Mat, Rect},
	imgproc,
};
use std::{sync::atomic::Ordering, thread};

fn basic_match(result: &mut Mat, frame: &Mat, template: &Mat) -> Result<f64> {
	let mut max_val: f64 = 0.0;
	imgproc::match_template(
		frame,
		template,
		result,
		imgproc::TM_CCOEFF_NORMED,
		&core::no_array(),
	)
	.wrap_err("template matching failed")?;
	core::min_max_loc(
		result,
		None,
		Some(&mut max_val),
		None,
		None,
		&core::no_array(),
	)
	.wrap_err("calculating global maximum value failed")?;
	Ok(max_val)
}

fn process_frame(
	bounds: Option<&Rect>,
	result: &mut Mat,
	frame: Mat,
	pos_templates: &[Mat],
	neg_templates: &[Mat],
	pos_threshold: Option<f64>,
	neg_threshold: Option<f64>,
) -> Result<(bool, f64, f64)> {
	let frame = match bounds {
		Some(bounds) => {
			Mat::roi(&frame, *bounds).wrap_err_with(|| format!("invalid roi: {bounds:?}"))?
		}
		None => frame,
	};

	let mut pos: f64 = 0.0;
	let mut neg: f64 = 0.0;
	let mut matched = false;
	for template in pos_templates {
		pos = pos.max(basic_match(result, &frame, template)?);
		match (pos_threshold, neg_threshold) {
			(Some(pos_threshold), None) if pos >= pos_threshold => return Ok((true, pos, 0.0)),
			(Some(pos_threshold), _) if pos >= pos_threshold => {
				matched = true;
				break;
			}
			_ => {}
		}
	}

	if matched || pos_threshold.is_none() {
		for template in neg_templates {
			neg = neg.max(basic_match(result, &frame, template)?);
			match neg_threshold {
				Some(neg_threshold) if neg >= neg_threshold => return Ok((false, pos, neg)),
				_ => {}
			}
		}
	}

	Ok((matched, pos, neg))
}

pub fn worker_thread(
	bounds: Option<Rect>,
	pos_templates: Vec<Mat>,
	neg_templates: Vec<Mat>,
	pos_threshold: Option<f64>,
	neg_threshold: Option<f64>,
	frame_receiver: FrameReceiver,
	result_sender: MatchedFrameSender,
) -> Result<()> {
	let mut result = Mat::default();
	for Frame { index, frame } in frame_receiver.iter() {
		let (result, ..) = process_frame(
			bounds.as_ref(),
			&mut result,
			frame,
			&pos_templates,
			&neg_templates,
			pos_threshold,
			neg_threshold,
		)
		.wrap_err_with(|| format!("failed to process frame {index} on cpu"))?;
		if result {
			result_sender.send(index).map_err(|_| {
				eyre!("failed to send result for frame {index} back to main thread")
			})?;
		}
		FRAMES_PROCESSED.fetch_add(1, Ordering::Relaxed);
	}
	Ok(())
}

pub fn spawn_threads(
	max_threads: Option<usize>,
	bounds: Option<Rect>,
	pos_templates: &[Mat],
	neg_templates: &[Mat],
	pos_threshold: Option<f64>,
	neg_threshold: Option<f64>,
	frame_receiver: FrameReceiver,
	result_sender: MatchedFrameSender,
) -> Result<()> {
	let core_ids = core_affinity::get_core_ids().wrap_err("failed to get CPU core IDs")?;
	for id in core_ids
		.iter()
		.copied()
		.take(max_threads.unwrap_or(usize::MAX))
	{
		let pos_templates = pos_templates.to_vec();
		let neg_templates = neg_templates.to_vec();
		let frame_receiver = frame_receiver.clone();
		let result_sender = result_sender.clone();
		thread::Builder::new()
			.name(format!("cpu worker core {}", id.id))
			.spawn(move || {
				if !core_affinity::set_for_current(id) {
					eprintln!("failed to set thread affinity for core {}", id.id);
				}
				worker_thread(
					bounds,
					pos_templates,
					neg_templates,
					pos_threshold,
					neg_threshold,
					frame_receiver,
					result_sender,
				)
				.expect("cpu worker thread errored");
			})
			.wrap_err_with(|| format!("failed to spawn cpu worker on core {}", id.id))?;
	}
	Ok(())
}
