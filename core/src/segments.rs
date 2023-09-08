use color_eyre::eyre::{Result, WrapErr};
use opencv::videoio::{VideoCapture, VideoCaptureTraitConst, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT};

pub type TimeRange = (f64, f64);

pub fn frames_to_segments<ExceedingFrames>(
	padding: f64,
	capture: &VideoCapture,
	exceeding_frames: ExceedingFrames,
) -> Result<Vec<TimeRange>>
where
	ExceedingFrames: AsRef<[usize]>,
{
	frames_to_segments_impl(padding, capture, exceeding_frames.as_ref())
}

fn frames_to_segments_impl(
	padding: f64,
	capture: &VideoCapture,
	exceeding_frames: &[usize],
) -> Result<Vec<TimeRange>> {
	let fps = capture
		.get(CAP_PROP_FPS)
		.wrap_err("failed to read fps property from video")?;
	let mut time_ranges = Vec::new();
	let mut start_frame = exceeding_frames[0];
	let mut end_frame = exceeding_frames[0];

	for i in 1..exceeding_frames.len() {
		if exceeding_frames[i] - exceeding_frames[i - 1] > 1 {
			let start_time = (start_frame as f64 / fps) - padding;
			let end_time = (end_frame as f64 / fps) + padding;
			time_ranges.push((start_time, end_time));

			start_frame = exceeding_frames[i];
		}
		end_frame = exceeding_frames[i];
	}

	// Add the last range if it was continuous
	let start_time = (start_frame as f64 / fps) - padding;
	let end_time = (end_frame as f64 / fps) + padding;
	time_ranges.push((start_time, end_time));

	let mut merged_time_ranges = Vec::new();
	let mut current_range = time_ranges[0];

	for &(start, end) in &time_ranges[1..] {
		if start <= current_range.1 {
			current_range.1 = current_range.1.max(end);
		} else {
			merged_time_ranges.push(current_range);
			current_range = (start, end);
		}
	}

	merged_time_ranges.push(current_range);

	let mut non_matching_time_ranges = Vec::new();
	let mut previous_end_time = 0.0;

	for (start_time, end_time) in merged_time_ranges.iter().copied() {
		let gap_duration = start_time - previous_end_time;
		if gap_duration > 0.0 {
			non_matching_time_ranges.push((previous_end_time, start_time));
		}
		previous_end_time = end_time;
	}

	// Calculate the total video duration
	let total_frames = capture
		.get(CAP_PROP_FRAME_COUNT)
		.wrap_err("failed to get frame count property from video")?;
	let total_duration = total_frames / fps;

	// Include the last non-matching time range if needed
	if total_duration - previous_end_time > 0.0 {
		non_matching_time_ranges.push((previous_end_time, total_duration));
	}

	Ok(non_matching_time_ranges)
}
