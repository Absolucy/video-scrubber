use crate::segments::TimeRange;
use color_eyre::eyre::{ContextCompat, Result, WrapErr};
use ffmpeg_next as ffmpeg;
use std::path::Path;

pub fn splice_video<Input, Output, Segments>(
	input: Input,
	output: Output,
	segments: Segments,
) -> Result<()>
where
	Input: AsRef<Path>,
	Output: AsRef<Path>,
	Segments: AsRef<[TimeRange]>,
{
	splice_video_impl(input.as_ref(), output.as_ref(), segments.as_ref())
}

fn splice_video_impl(input: &Path, output: &Path, segments: &[TimeRange]) -> Result<()> {
	ffmpeg::init().wrap_err("failed to initialize ffmpeg")?;

	let mut ictx = ffmpeg::format::input(&input)
		.wrap_err_with(|| format!("failed to open input file at {}", input.display()))?;
	let mut octx = ffmpeg::format::output(&output)
		.wrap_err_with(|| format!("failed to open output file at {}", output.display()))?;

	for (index, istream) in ictx.streams().enumerate() {
		let input_parameters = istream.parameters();
		let codec_id = input_parameters.id();
		let codec = ffmpeg::encoder::find(codec_id)
			.wrap_err_with(|| format!("failed to find codec id {codec_id:?}"))?;
		let mut ostream = octx.add_stream(codec).wrap_err_with(|| {
			format!("failed to add stream for stream {index} with codec id {codec_id:?}")
		})?;
		ostream.set_time_base(istream.time_base());
		ostream.set_parameters(input_parameters);
	}

	octx.write_header()
		.wrap_err("failed to write output header")?;

	let mut total_offset_pts = 0;
	let mut total_offset_dts = 0;

	let mut segment_idx = 0;
	let mut current_segment = segments.first().copied();

	for (stream, mut packet) in ictx.packets() {
		let (start_time, end_time) = match current_segment {
			Some(times) => times,
			None => break,
		};
		let time_base = f64::from(stream.time_base());

		let pts = packet.pts().wrap_err("invalid pts")?;
		let dts = packet.dts().wrap_err("invalid dts")?;

		let packet_pts = (pts as f64) * time_base;
		let packet_dts = (dts as f64) * time_base;

		// Skip packets that are before the current segment.
		if packet_pts < start_time || packet_dts < start_time {
			continue;
		}

		if packet_pts >= end_time || packet_dts >= end_time {
			segment_idx += 1;
			current_segment = segments.get(segment_idx).copied();
			match current_segment {
				Some((next_start, next_end)) => {
					total_offset_pts += ((next_start - end_time) / time_base) as i64;
					total_offset_dts = total_offset_pts;
					if packet_pts < next_start || packet_pts >= next_end {
						continue;
					}
				}
				_ => break,
			}
		}

		// Adjust the PTS and DTS for continuous playback.
		let new_pts = pts - total_offset_pts;
		let new_dts = dts - total_offset_dts;
		packet.set_pts(Some(new_pts));
		packet.set_dts(Some(new_dts));

		packet
			.write_interleaved(&mut octx)
			.wrap_err("failed to write interleaved packet")?;
	}

	octx.write_trailer()
		.wrap_err("failed to write output trailer")?;
	Ok(())
}
