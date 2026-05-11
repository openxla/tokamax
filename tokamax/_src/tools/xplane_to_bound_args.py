# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Extracts bound argument specifications (op specs) from an XPlane proto file.

This script reads an XPlane proto file (XSpace), extracts the HLO modules
contained within it, and then extracts unique bound argument specifications
(op specs) from those modules. The extracted op specs are written to an output
file.

Example usage:
  python tokamax/_src/tools/xplane_to_bound_args.py \
    --xplane_file=/tmp/xspace.pb \
    --output_file=/tmp/bound_args.txt

Flags:
  --xplane_file: Path to the input XPlane proto file. Required.
  --output_file: Path to the output file where op specs will be written. Required.
"""
import sys
from absl import app
from absl import flags
from tokamax.google import hlo_utils
from tensorflow.compiler.xla.service import hlo_pb2
from tensorflow.tsl.profiler.protobuf import xplane_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string("xplane_file", None, "Path to XPlane proto file.")
flags.DEFINE_string("output_file", None, "Output file name.")

flags.mark_flag_as_required("xplane_file")
flags.mark_flag_as_required("output_file")


def get_hlo_protos_from_xspace(xspace: xplane_pb2.XSpace):
  hlo_protos = []
  for plane in xspace.planes:
    if plane.name == "/host:metadata":
      hlo_metadata_id = None
      for metadata_id, stat_meta in plane.stat_metadata.items():
        if stat_meta.name == "Hlo Proto":
          hlo_metadata_id = metadata_id
          break

      if hlo_metadata_id is None:
        continue

      for event_meta in plane.event_metadata.values():
        for stat in event_meta.stats:
          if stat.metadata_id == hlo_metadata_id:
            if stat.HasField("bytes_value"):
              hlo_proto = hlo_pb2.HloProto()
              hlo_proto.ParseFromString(stat.bytes_value)
              if not hlo_proto.hlo_module.name:
                hlo_proto.hlo_module.name = event_meta.name or "unknown_module"
              hlo_protos.append(hlo_proto)
  return hlo_protos


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  try:
    print(f"Reading XPlane data from {FLAGS.xplane_file}...")
    xspace = xplane_pb2.XSpace()
    with open(FLAGS.xplane_file, "rb") as f:
      xspace.ParseFromString(f.read())
  except Exception as e:
    print(f"Error reading XPlane data: {e}", file=sys.stderr)
    sys.exit(1)

  hlo_protos = get_hlo_protos_from_xspace(xspace)
  if not hlo_protos:
    print("No HLO modules found in the XPlane data.", file=sys.stderr)
    sys.exit(1)

  print(f"Found {len(hlo_protos)} HLO modules.")

  all_opspecs = []
  for hlo_proto in hlo_protos:
    print(f"Processing module: {hlo_proto.hlo_module.name}")
    opspecs = hlo_utils.get_opspecs(hlo_proto.hlo_module)
    all_opspecs.extend(opspecs)

  seen_keys = set()
  unique_opspecs = []
  for opspec in all_opspecs:
    # The chosen config is serialized into the HLO - remove it here.
    opspec = opspec.replace(op=opspec.op.replace(config=None))
    key = opspec.autotuning_cache_key
    if (opspec.op.__class__.__name__, key) not in seen_keys:
      seen_keys.add((opspec.op.__class__.__name__, key))
      unique_opspecs.append(opspec)
  all_opspecs = unique_opspecs

  print(f"Found {len(all_opspecs)} op specs.")

  with open(FLAGS.output_file, "w") as f:
    for opspec in all_opspecs:
      f.write(str(opspec) + "\n")

  print(f"Results written to {FLAGS.output_file}")


if __name__ == "__main__":
  app.run(main)
