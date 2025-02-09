import trimesh
import numpy as np
from PIL import Image
import cv2
import io
import os
import ffmpeg

from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def vis_meshes(*meshes, show=False):
    """Visualize meshes. Return as numpy array unless show is True, in which case show"""
    for mesh in meshes:
        assert isinstance(
            mesh, trimesh.Trimesh
        ), "All meshes for visualisation must be Trimesh."

    sce = trimesh.Scene(geometry=meshes)
    if show:
        sce.show()

    else:
        data = sce.save_image(resolution=(200, 200), visible=False)
        image = np.array(Image.open(io.BytesIO(data)))
        return image


def vis_meshes_pyplot(*meshlists, titles: list = None, extra_text="", colours=None):
    for meshlist in meshlists:
        if isinstance(meshlist, trimesh.Trimesh):
            meshlist = [meshlist]

        for mesh in meshlist:
            assert isinstance(
                mesh, trimesh.Trimesh
            ), "All meshes for visualisation must be Trimesh."

    fig = plt.figure()

    for n, meshlist in enumerate(meshlists):
        if isinstance(meshlist, trimesh.Trimesh):
            meshlist = [meshlist]

        ax = fig.add_subplot(1, len(meshlists), n + 1, projection="3d")

        for mesh in meshlist:
            X, Y, Z = mesh.vertices.T
            tris = Triangulation(X, Y, triangles=mesh.faces)
            trisurf = ax.plot_trisurf(tris, Z, alpha=1 if len(meshlist) == 1 else 0.5)

            # Assume single mesh
            if colours is not None:
                c = colours[n]
                if c is None:
                    continue
                if len(meshlist) > 1:
                    raise NotImplementedError(
                        "Cannot take colours argument for multiple meshes"
                    )
                trisurf.set_fc(c)

        if titles is not None and isinstance(titles, list):
            ax.set_title(titles[n])

        set_axes_equal(ax)

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    fig.text(0.5, 0.1, extra_text, ha="center")
    # plt.show()

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=200)
    io_buf.seek(0)
    data = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    plt.close(fig)

    return data


def vis_pcl_pyplot(verts: list, c=None, titles: list = None, vmax=None, vmin=None):
    """Visualize pointclouds with colours"""

    fig = plt.figure()

    N = len(verts)
    for n in range(N):
        ax = fig.add_subplot(1, N, n + 1, projection="3d")
        _c = None if c is None else c[n]
        ax.scatter(*verts[n].T, c=_c, s=0.2, cmap="bwr", vmax=vmax, vmin=vmin)
        set_axes_equal(ax)
        if titles is not None:
            ax.set_title(titles[n])

    # plt.show()

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", dpi=200)
    io_buf.seek(0)
    data = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    plt.close(fig)

    return data


def frames_to_video(directory, ext="png", delete_images=True, out_fname="out"):

    out_file = os.path.join(directory, f"{out_fname}.mp4")

    (
        ffmpeg.input(
            os.path.join(directory, f"*.{ext}"), pattern_type="glob", framerate=10
        )
        .output(out_file, format="mp4", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )

    if delete_images:
        for f in os.listdir(directory):
            if f.endswith(f".{ext}"):
                os.remove(os.path.join(directory, f))

    return out_file


def visualize_classes(arr, seed=7):
    """Given an array of H x W x C probability heatmap
    return a coloured H x W image corresponding to the most likely of the C classes given per pixel.
     Colours generated by seed"""
    H, W, C = arr.shape
    # Generate random colours
    np.random.seed(seed)
    cols = np.random.randint(0, 255, size=(C, 3))
    cols[0] = 0

    classes = np.argmax(arr, axis=-1)
    out = np.zeros((H, W, 3)).astype(np.uint8)
    for i in range(C):
        out[classes == i] = cols[i]

    return out


def visualize_classes_argmax(classes, C=21, seed=7):
    """Given an array of H x W of classes, in range 0 <= i <= C
    return a coloured H x W image corresponding to the class C classes given per pixel.
     Colours generated by seed"""
    H, W = classes.shape
    # Generate random colours
    np.random.seed(seed)
    cols = np.random.randint(0, 255, size=(C, 3))
    cols[0] = 0

    out = np.zeros((H, W, 3)).astype(np.uint8)
    for i in range(C):
        out[classes == i] = cols[i]

    return out
