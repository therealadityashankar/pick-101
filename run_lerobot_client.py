"""Launcher for lerobot's async_inference robot_client.

Works around two issues with running the client directly:
  1. lerobot.robots.__init__ never imports the concrete robot modules, so the
     --robot.type choices come up empty ("invalid choice ... choose from ").
     Importing the SO-follower config here runs its @register_subclass
     decorators and populates the registry.
  2. Must run under the venv python directly (not `uv run`, which re-syncs and
     downgrades protobuf back below the version the gRPC stubs need).

Usage (note: venv python, NOT uv run):
    .venv/bin/python run_lerobot_client.py \
        --server_address=216.243.220.169:18564 \
        --robot.type=so101_follower \
        --robot.port=/dev/tty.usbmodem5A680089441 \
        --robot.id=so101 \
        --robot.cameras="{ laptop: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30} }" \
        --task="pick up the wooden block" \
        --policy_type=molmoact2 \
        --pretrained_name_or_path=lerobot/MolmoAct2-SO100_101-LeRobot \
        --policy_device=cuda \
        --actions_per_chunk=50 \
        --chunk_size_threshold=0.5 \
        --aggregate_fn_name=weighted_average \
        --debug_visualize_queue_size=True
"""

# Register the SO-100/SO-101 follower robot types before argparse builds choices
import lerobot.robots.so_follower.config_so_follower  # noqa: F401

from lerobot.async_inference.robot_client import async_client
from lerobot.utils.import_utils import register_third_party_plugins

if __name__ == "__main__":
    register_third_party_plugins()
    async_client()
