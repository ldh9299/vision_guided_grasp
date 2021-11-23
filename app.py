import torch
import argparse
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.axislayer import AxisLayer
from manotorch.anchorlayer import AnchorLayer
from manotorch.utils.visutils import display_hand_matplot, display_hand_pyrender, display_hand_open3d


def main(args):
    # Initialize MANO layer
    ncomps = 45
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=True,
        side="right",
        center_idx=None,
        mano_assets_root="assets/mano",
        flat_hand_mean=not args.flat_hand_mean,
        ncomps=ncomps,
    )
    axis_layer = AxisLayer()
    anchor_layer = AnchorLayer(anchor_root="assets/anchor")

    batch_size = 1
    torch.manual_seed(0)
    # Generate random shape parameters
    random_shape = torch.rand(batch_size, 10) * 0
    # Generate random pose parameters, including 3 values for global axis-angle rotation
    random_pose = torch.rand(batch_size, 3 + ncomps) * 0

    # print(random_shape, random_pose)
    betas_ = torch.tensor([[0.0021263046711297224, 0.00223549247827398, -0.00030514961293518, 0.00044706099963705146,
                            0.00016999565020175557, -0.0012500970520221797, -0.0006832669036269286,
                            0.0009646884338019807, -0.00031103207561103, -0.00032905848501462825]])
    pose_ = torch.tensor([[0., 0., 0., 0.11167872, -0.04289217, 0.41644184,
                           0.10881133, 0.06598568, 0.75622001, -0.09639297, 0.09091566, 0.18845929,
                           -0.11809504, -0.05094385, 0.5295845, -0.14369841, -0.0552417, 0.70485714,
                           -0.01918292, 0.09233685, 0.33791352, -0.45703298, 0.19628395, 0.62545753,
                           -0.21465238, 0.06599829, 0.50689421, -0.36972436, 0.06034463, 0.07949023,
                           -0.14186969, 0.08585263, 0.63552826, -0.30334159, 0.05788098, 0.63138921,
                           -0.17612089, 0.13209308, 0.37335458, 0.85096428, -0.27692274, 0.09154807,
                           -0.49983944, -0.02655647, -0.05288088, 0.53555915, -0.04596104, 0.27735802]])
    # mano_results: MANOOutput = mano_layer(pose_, betas_)
    mano_results: MANOOutput = mano_layer(random_pose, random_shape)
    """
    MANOOutput = namedtuple(
        "MANOOutput",
        [
            "verts",
            "joints",
            "center_idx",
            "center_joint",
            "full_poses",
            "betas",
            "transforms_abs",
        ],
    )
    """
    print(mano_results.full_poses)
    verts = mano_results.verts
    joints = mano_results.joints
    # print('joints', joints)
    transforms_abs = mano_results.transforms_abs
    # print('transforms_abs', transforms_abs)

    anchors = anchor_layer(verts)
    bul_axes = axis_layer(joints, transforms_abs)

    if args.display == "pyrender":
        display_hand_pyrender(mano_results, mano_layer.th_faces, bul_axes=bul_axes, anchors=anchors)
    elif args.display == "open3d":
        display_hand_open3d(mano_results, mano_layer.th_faces)
    elif args.display == "matplot":
        display_hand_matplot(mano_results, mano_layer.th_faces)
    else:
        print("Unknown display")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat_hand_mean", action="store_true", help="Use flat hand as mean")
    parser.add_argument("--display", choices=["matplot", "pyrender", "open3d"], default="pyrender", type=str)
    main(parser.parse_args())
