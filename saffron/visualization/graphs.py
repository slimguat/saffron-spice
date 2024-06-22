from configparser import NoOptionError
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

import numpy as np

from astropy.visualization import (
    SqrtStretch,
    AsymmetricPercentileInterval,
    ImageNormalize,
)

from ..utils.utils import gen_velocity


def _plot_window_Miho(
    spectrum_axis: np.array,
    window: np.ndarray,
    paramlist: np.ndarray,
    quentity: list,
    convlist=None,
    suptitle="",
    window_size: np.ndarray = np.array([[0, -1], [0, -1]]),
    t: int = 0,
    segmentation=None,
    save=False,
    filename="./imgs/res.jpg",
    quite_sun: np.ndarray = np.array([0, -1, 0, -1]),
    min_I=None,
    max_I=None,
    min_x=-80,
    max_x=+80,
    min_s=None,
    max_s=None,
    min_B=None,
    max_B=None,
    visualize_saturation=True,
    **kwargs
):
    w = window
    q = quentity
    p = paramlist
    ws = window_size
    sa = spectrum_axis
    cmap_doppler = "RdBu_r"
    if "raster" in kwargs:
        raster = kwargs["raster"]

        All_ang_lat = raster.celestial.data.lat.deg
        All_ang_lon = raster.celestial.data.lon.deg

        All_ang_lat2 = All_ang_lat.copy()
        All_ang_lon2 = All_ang_lon.copy()

        All_ang_lon2[All_ang_lon <= 180] = All_ang_lon[All_ang_lon <= 180] * 3600
        All_ang_lon2[All_ang_lon > 180] = (All_ang_lon[All_ang_lon > 180] - 360) * 3600

        All_ang_lat2[All_ang_lat <= 180] = All_ang_lat[All_ang_lat <= 180] * 3600
        All_ang_lat2[All_ang_lat > 180] = (All_ang_lat[All_ang_lat > 180]) * 3600

        All_ang_lat = All_ang_lat2.copy()
        All_ang_lon = All_ang_lon2.copy()

    qs2 = quite_sun.copy()

    quite_sun = quite_sun.copy()
    # quite_sun[0] = quite_sun[0] - window_size[1,0]
    # quite_sun[2] = quite_sun[2] - window_size[0,0]

    if quite_sun[1] == -1:
        if window_size[1, 1] == -1:
            quite_sun[1] = window.shape[3]  # - window_size[1,0]
        else:
            quite_sun[1] = window_size[1, 1]  # - window_size[1,0]
    else:
        pass  # quite_sun[1] = quite_sun[1] - window_size[1,0]
    if quite_sun[3] == -1:
        if window_size[0, 1] == -1:
            quite_sun[3] = window.shape[2]  # - window_size[0,0]
        else:
            quite_sun[3] = window_size[0, 1]  # - window_size[0,0]
    else:
        quite_sun[3] = quite_sun[3]  # - window_size[0,0]

    if "raster" in kwargs:
        qs = qs2.copy()
        QS_up_leftx = (
            All_ang_lon[qs[3], qs[0]]
            + (All_ang_lon[qs[3], qs[0]] - (All_ang_lon[qs[3], qs[0] + 1])) / 2
        )
        QS_up_lefty = (
            All_ang_lat[qs[3], qs[0]]
            - (All_ang_lat[qs[3], qs[0]] - (All_ang_lat[qs[3] - 1, qs[0]])) / 2
        )
        QS_up_rightx = (
            All_ang_lon[qs[3], qs[1]]
            - (All_ang_lon[qs[3], qs[1]] - (All_ang_lon[qs[3], qs[1] - 1])) / 2
        )
        QS_up_righty = (
            All_ang_lat[qs[3], qs[1]]
            - (All_ang_lat[qs[3], qs[1]] - (All_ang_lat[qs[3] - 1, qs[1]])) / 2
        )
        QS_down_rightx = (
            All_ang_lon[qs[2], qs[1]]
            - (All_ang_lon[qs[2], qs[1]] - (All_ang_lon[qs[2], qs[1] - 1])) / 2
        )
        QS_down_righty = (
            All_ang_lat[qs[2], qs[1]]
            + (All_ang_lat[qs[2], qs[1]] - (All_ang_lat[qs[2] + 1, qs[1]])) / 2
        )
        QS_down_leftx = (
            All_ang_lon[qs[2], qs[0]]
            + (All_ang_lon[qs[2], qs[0]] - (All_ang_lon[qs[2], qs[0] + 1])) / 2
        )
        QS_down_lefty = (
            All_ang_lat[qs[2], qs[0]]
            + (All_ang_lat[qs[2], qs[0]] - (All_ang_lat[qs[2] + 1, qs[0]])) / 2
        )
        # print(qs[3],qs[0],QS_up_leftx)
        ws = window_size
        WS_up_leftx = (
            All_ang_lon[ws[0, 1], ws[1, 0]]
            + (All_ang_lon[ws[0, 1], ws[1, 0]] - (All_ang_lon[ws[0, 1], ws[1, 0] + 1]))
            / 2
        )
        WS_up_lefty = (
            All_ang_lat[ws[0, 1], ws[1, 0]]
            - (All_ang_lat[ws[0, 1], ws[1, 0]] - (All_ang_lat[ws[0, 1] - 1, ws[1, 0]]))
            / 2
        )
        WS_up_rightx = (
            All_ang_lon[ws[0, 1], ws[1, 1]]
            - (All_ang_lon[ws[0, 1], ws[1, 1]] - (All_ang_lon[ws[0, 1], ws[1, 1] - 1]))
            / 2
        )
        WS_up_righty = (
            All_ang_lat[ws[0, 1], ws[1, 1]]
            - (All_ang_lat[ws[0, 1], ws[1, 1]] - (All_ang_lat[ws[0, 1] - 1, ws[1, 1]]))
            / 2
        )
        WS_down_rightx = (
            All_ang_lon[ws[0, 0], ws[1, 1]]
            - (All_ang_lon[ws[0, 0], ws[1, 1]] - (All_ang_lon[ws[0, 0], ws[1, 1] - 1]))
            / 2
        )
        WS_down_righty = (
            All_ang_lat[ws[0, 0], ws[1, 1]]
            + (All_ang_lat[ws[0, 0], ws[1, 1]] - (All_ang_lat[ws[0, 0] + 1, ws[1, 1]]))
            / 2
        )
        WS_down_leftx = (
            All_ang_lon[ws[0, 0], ws[1, 0]]
            + (All_ang_lon[ws[0, 0], ws[1, 0]] - (All_ang_lon[ws[0, 0], ws[1, 0] + 1]))
            / 2
        )
        WS_down_lefty = (
            All_ang_lat[ws[0, 0], ws[1, 0]]
            + (All_ang_lat[ws[0, 0], ws[1, 0]] - (All_ang_lat[ws[0, 0] + 1, ws[1, 0]]))
            / 2
        )

    def sub_q(Q: list) -> list:
        sub_q = []
        i_b1 = 0
        i_b2 = 0
        for i in range(len(Q)):
            if Q[i] == "B":
                i_b2 = i + 1
                sub_q.append([i_b1, i_b2])
                i_b1 = i_b2
        return sub_q

    _q = sub_q(q)
    _c = -1
    _nl = int((len(q) - len(_q)) / 3) + len(_q) + 1
    # conv_c = 0
    mean_pos = []
    plt.rcParams.update({"font.size": 22})
    fig, axis = plt.subplots(_nl, 3, figsize=(24, 8.2 * _nl), constrained_layout=True)

    for i in range(len(q)):
        if q[i] == "I":
            _c += 1
            minmax = {"interval": AsymmetricPercentileInterval(1, 99)}
            if type(max_I) != type(None) and type(min_I) != type(None):
                minmax = {"vmin": min_I, "vmax": max_I}

            try:
                norm = ImageNormalize(p[i, t], **minmax, stretch=SqrtStretch())
            except:
                pass

            try:
                if "raster" in kwargs:
                    im = axis[_c, 0].pcolormesh(
                        All_ang_lon, All_ang_lat, p[i, t], norm=norm, cmap="magma"
                    )
                else:
                    im = axis[_c, 0].imshow(
                        p[i, t], aspect="auto", origin="lower", norm=norm, cmap="magma"
                    )
            except:
                if "raster" in kwargs:
                    im = axis[_c, 0].pcolormesh(
                        All_ang_lon, All_ang_lat, p[i, t], cmap="magma"
                    )
                else:
                    im = axis[_c, 0].imshow(
                        p[i, t], aspect="auto", origin="lower", cmap="magma"
                    )

            axis[_c, 0].set_title(
                "Intensity ($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)"
            )
            axis[_c, 0].set_xlabel("Helioprojective longitude \n (arcsec)")
            axis[_c, 0].set_ylabel("Helioprojective latitude \n (arcsec)")
            axis[_c, 0].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 0], extend=("both" if visualize_saturation else None)
            )

        if q[i] == "x":
            mean_x = np.nanmean(
                p[
                    i,
                    t,
                    quite_sun[2] - ws[0, 0] : quite_sun[3] + ws[0, 0],
                    quite_sun[0] - ws[1, 0] : quite_sun[1] - ws[1, 0],
                ]
            )
            # print(quite_sun)
            # print(ws)
            # print([i,t,quite_sun[2]-ws[0,1],quite_sun[3]+ws[0,1],quite_sun[0]-ws[1,0],quite_sun[1]-ws[1,0]])
            cmap = plt.get_cmap(cmap_doppler).copy()
            (
                cmap.set_extremes(under="yellow", over="green")
                if visualize_saturation
                else 0
            )
            if "raster" in kwargs:
                # print("Calculated mean: ", mean_x)
                im = axis[_c, 1].pcolormesh(
                    All_ang_lon,
                    All_ang_lat,
                    (p[i, t] - mean_x) / mean_x * 3 * 10**5,
                    vmin=min_x,
                    vmax=max_x,
                    cmap=cmap,
                )
            else:
                im = axis[_c, 1].imshow(
                    (p[i, t] - mean_x) / mean_x * 3 * 10**5,
                    origin="lower",
                    vmin=min_x,
                    vmax=max_x,
                    aspect="auto",
                    cmap=cmap,
                )
            axis[_c, 1].set_title("Doppler (km/s)")
            axis[_c, 1].set_xlabel("Helioprojective longitude \n (arcsec)")
            axis[_c, 1].set_ylabel("Helioprojective latitude \n (arcsec)")
            axis[_c, 1].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 1], extend=("both" if visualize_saturation else None)
            )
            mean_pos.append(np.nanmean(p[i, t]))
        if q[i] == "s":
            cmap = plt.get_cmap("hot").copy()
            (
                cmap.set_extremes(under="green", over="violet")
                if visualize_saturation
                else 0
            )
            if "raster" in kwargs:
                im = axis[_c, 2].pcolormesh(
                    All_ang_lon, All_ang_lat, p[i, t], cmap=cmap, vmax=max_s, vmin=min_s
                )
            else:
                im = axis[_c, 2].imshow(
                    p[i, t], aspect="auto", origin="lower", cmap=cmap
                )

            axis[_c, 2].set_title("$\sigma$ ($\AA$)")
            axis[_c, 2].set_xlabel("Helioprojective longitude \n (arcsec)")
            axis[_c, 2].set_ylabel("Helioprojective latitude \n (arcsec)")
            axis[_c, 2].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 2], extend=("both" if visualize_saturation else None)
            )
        if q[i] == "B":
            _c += 1
            minmax = (
                {"interval": AsymmetricPercentileInterval(1, 99)}
                if type(max_B) == type(None)
                else {"vmin": min_B, "vmax": max_B}
            )
            good_data = True
            try:
                norm = ImageNormalize(p[i, t], **minmax, stretch=SqrtStretch())
            except:
                print("this data contains nans everywhere")
                good_data = False
            if "raster" in kwargs:
                im = axis[_c, 0].pcolormesh(
                    All_ang_lon,
                    All_ang_lat,
                    p[i, t],
                    norm=norm if good_data else None,
                    cmap="magma",
                )
            else:
                im = axis[_c, 0].imshow(
                    p[i, t],
                    aspect="auto",
                    origin="lower",
                    norm=norm if good_data else None,
                    cmap="magma",
                )
            axis[_c, 0].set_title(
                "Background Intensity\n($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)"
            )
            axis[_c, 0].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 0], extend=("both" if visualize_saturation else None)
            )
            axis[_c, 0].set_xlabel("Helioprojective longitude \n (arcsec)")
            axis[_c, 0].set_ylabel("Helioprojective latitude \n (arcsec)")
            axis[_c, 2].remove()
            if type(convlist) == type(None):
                axis[_c, 1].remove()
            else:
                data_conv = convlist[t, :, :].copy()
                # designing colormap -> color choice and their number
                vals = (np.unique(data_conv)).astype(np.int8)
                # print(vals)
                # print("vals",vals)
                N = len(vals)
                # print(int(N**(1/3)),N**(1/3),'int(N**(1/3)),N**(1/3)')
                n = int(N ** (1 / 3)) + (0 if int(N ** (1 / 3)) == N ** (1 / 3) else 1)
                yn = N ** (1 / 3)
                # print("N,n",N,n)
                col_dict = {}
                i = 0
                for b in range(n):
                    if i == N:
                        break
                    for g in range(n):
                        if i == N:
                            break
                        for r in range(n):
                            if i == N:
                                break
                            col_dict[vals[i]] = (
                                1 - r / max(n - 1, 1),
                                1 - g / max(n - 1, 1),
                                1 - b / max(n - 1, 1),
                            )
                            i += 1
                # print("col_dict",col_dict)
                # designing colormap -> constructing the cmap
                cmap = ListedColormap([col_dict[x] for x in col_dict.keys()])

                # In case you want the labels to show other than the numbers put the list of these labels here
                labels = vals.copy()
                len_lab = len(labels)

                # prepare for the normalizer
                norm_bins = np.sort([*col_dict.keys()]) + 0.5
                # print("sort",norm_bins)
                norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
                # print("insert", norm_bins)
                # Make normalizer and formatter
                norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
                fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

                if "raster" in kwargs:
                    im = axis[_c, 1].pcolormesh(
                        All_ang_lon, All_ang_lat, data_conv, cmap=cmap, norm=norm
                    )
                else:
                    im = axis[_c, 1].imshow(
                        data_conv,
                        aspect="auto",
                        origin="lower",
                        cmap=cmap,
                        norm=norm,
                    )
                axis[_c, 1].set_xlabel("Helioprojective longitude \n (arcsec)")
                axis[_c, 1].set_ylabel("Helioprojective latitude \n (arcsec)")
                axis[_c, 1].set_title("convolution level map")
                axis[_c, 1].set_aspect("equal")

                diff = norm_bins[1:] - norm_bins[:-1]
                tickz = norm_bins[:-1] + diff / 2
                plt.colorbar(im, ax=axis[_c, 1], ticks=tickz)
                # conv_c  += 1

    # plt.rcParams.update({'font.size': 16})
    if "raster" in kwargs:

        mn = np.nanmean(
            w[:, :, :, :], axis=(0, 1)  # ws[0,0]:ws[0,1],  # ws[1,0]:ws[1,1]
        )
        minmax = {"interval": AsymmetricPercentileInterval(1, 99)}
        if type(max_I) != type(None) and type(min_I) != type(None):
            minmax = {"vmin": min_I, "vmax": max_I}

        norm = ImageNormalize(mn, **minmax, stretch=SqrtStretch())
        im = axis[-1, 0].pcolormesh(
            All_ang_lon, All_ang_lat, mn, norm=norm, cmap="magma"
        )
        # axis[-1,0].set_aspect('auto')
        plt.colorbar(
            im, ax=axis[-1, 0], extend=("both" if visualize_saturation else None)
        )

        axis[-1, 0].plot(
            [QS_up_leftx, QS_up_rightx, QS_down_rightx, QS_down_leftx, QS_up_leftx],
            [QS_up_lefty, QS_up_righty, QS_down_righty, QS_down_lefty, QS_up_lefty],
            "o-",
            alpha=1,
            color=(0, 1, 0),
            lw=5,
            label="selected\nquite region",
        )
        axis[-1, 0].plot(
            [WS_up_leftx, WS_up_rightx, WS_down_rightx, WS_down_leftx, WS_up_leftx],
            [WS_up_lefty, WS_up_righty, WS_down_righty, WS_down_lefty, WS_up_lefty],
            "o-",
            alpha=1,
            color=(1, 0, 0),
            lw=3,
            label="Analysed region",
        )

    else:
        mn = np.nanmean(w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 1))
        norm = ImageNormalize(
            mn, AsymmetricPercentileInterval(1, 99), stretch=SqrtStretch()
        )

        im = axis[-1, 0].imshow(mn, norm=norm, cmap="magma", origin="lower")
        axis[-1, 0].set_xlabel("Helioprojective longitude \n (arcsec)")
        axis[-1, 0].set_ylabel("Helioprojective latitude \n (arcsec)")

        lims = axis[-1, 0].get_xlim(), axis[-1, 0].get_ylim()
        axis[-1, 0].set_aspect("auto")
        plt.colorbar(
            im, ax=axis[-1, 0], extend=("both" if visualize_saturation else None)
        )

        axis[-1, 0].axvspan(
            xmin=quite_sun[0],
            xmax=quite_sun[1],
            ymin=quite_sun[2] / (axis[-1, 0].get_ylim()[1] - axis[-1, 0].get_ylim()[0]),
            ymax=quite_sun[3] / (axis[-1, 0].get_ylim()[1] - axis[-1, 0].get_ylim()[0]),
            alpha=0.5,
            lw=2,
            color=(0, 1, 0),
            label="selected\nquite region",
        )

    axis[-1, 0].set_title("original averaged\n over spectrum")
    axis[-1, 0].set_xlabel("Helioprojective longitude \n (arcsec)")
    axis[-1, 0].set_ylabel("Helioprojective latitude \n (arcsec)")
    axis[-1, 1].step(
        sa,
        np.nanmean(w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 2, 3)),
    )

    # print(mean_pos)
    # print(sa)
    for x in mean_pos:
        axis[-1, 1].axvline(x, ls=":", label="line: {:02d}".format(mean_pos.index(x)))
    axis[-1, 1].legend()
    axis[-1, 1].set_title(
        "original average spectrum/\naverage line positions/\n segments"
    )

    axis[-1, 2].remove()
    fig.suptitle(suptitle + "\n" + raster.meta["DATE_SUN"][:-4], fontsize=20)
    if type(segmentation) != type(None):
        if len(segmentation.shape) != 1:
            for seg in segmentation:
                color = np.random.rand(3)
                color = 0.8 * color / np.sqrt(np.sum(color**2))
                axis[-1, 1].axvspan(seg[0], seg[1], alpha=0.5, color=color)
        else:
            seg = segmentation
            color = np.random.rand(3)
            color = 0.8 * color / np.sqrt(np.sum(color**2))
            axis[-1, 1].axvspan(seg[0], seg[1], alpha=0.5, color=color)
    for ax in axis.flatten():
        ax.ticklabel_format(useOffset=False)
        ax.ticklabel_format(useOffset=False)
    # fig.tight_layout()
    if save:
        plt.savefig(filename)

    return fig, axis, All_ang_lon, All_ang_lat


def plot_window_Miho(
    spectrum_axis: np.array,
    window: np.ndarray,
    paramlist: np.ndarray,
    quentity: list,
    convlist=None,
    suptitle="",
    window_size: np.ndarray = np.array([[0, -1], [0, -1]]),
    t: int = 0,
    segmentation=None,
    save=False,
    filename="./imgs/res.jpg",
    quite_sun: np.ndarray = np.array([0, -1, 0, -1]),
    min_I=None,
    max_I=None,
    min_x=-50,
    max_x=+50,
    min_s=None,
    max_s=None,
    min_B=None,
    max_B=None,
    visualize_saturation=True,
    **kwargs
):
    w = window
    q = quentity
    p = paramlist
    ws = window_size
    sa = spectrum_axis
    cmap_doppler = "twilight_shifted"
    if "raster" in kwargs:
        raster = kwargs["raster"]
        ang_lat = raster.celestial.data[
            ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]
        ].lat.deg
        ang_lon = raster.celestial.data[
            ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]
        ].lon.deg
        All_ang_lat = raster.celestial.data.lat.deg
        All_ang_lon = raster.celestial.data.lon.deg

        ang_lat2 = ang_lat.copy()
        ang_lon2 = ang_lon.copy()
        All_ang_lat2 = All_ang_lat.copy()
        All_ang_lon2 = All_ang_lon.copy()

        All_ang_lon2[All_ang_lon <= 180] = All_ang_lon[All_ang_lon <= 180] * 3600
        All_ang_lon2[All_ang_lon > 180] = (All_ang_lon[All_ang_lon > 180] - 360) * 3600

        ang_lon2[ang_lon <= 180] = ang_lon[ang_lon <= 180] * 3600
        ang_lon2[ang_lon > 180] = (ang_lon[ang_lon > 180] - 360) * 3600

        All_ang_lat2[All_ang_lat <= 180] = All_ang_lat[All_ang_lat <= 180] * 3600
        All_ang_lat2[All_ang_lat > 180] = (All_ang_lat[All_ang_lat > 180]) * 3600

        ang_lat2[ang_lat <= 180] = ang_lat[ang_lat <= 180] * 3600
        ang_lat2[ang_lat > 180] = (ang_lat[ang_lat > 180]) * 3600
        ang_lat = ang_lat2.copy()
        ang_lon = ang_lon2.copy()
        All_ang_lat = All_ang_lat2.copy()
        All_ang_lon = All_ang_lon2.copy()

    qs2 = quite_sun.copy()

    quite_sun = quite_sun.copy()
    # quite_sun[0] = quite_sun[0] - window_size[1,0]
    # quite_sun[2] = quite_sun[2] - window_size[0,0]

    if quite_sun[1] == -1:
        if window_size[1, 1] == -1:
            quite_sun[1] = window.shape[3]  # - window_size[1,0]
        else:
            quite_sun[1] = window_size[1, 1]  # - window_size[1,0]
    else:
        pass  # quite_sun[1] = quite_sun[1] - window_size[1,0]
    if quite_sun[3] == -1:
        if window_size[0, 1] == -1:
            quite_sun[3] = window.shape[2]  # - window_size[0,0]
        else:
            quite_sun[3] = window_size[0, 1]  # - window_size[0,0]
    else:
        quite_sun[3] = quite_sun[3]  # - window_size[0,0]

    if "raster" in kwargs:
        qs = qs2.copy()
        QS_up_leftx = (
            All_ang_lon[qs[3], qs[0]]
            + (All_ang_lon[qs[3], qs[0]] - (All_ang_lon[qs[3], qs[0] + 1])) / 2
        )
        QS_up_lefty = (
            All_ang_lat[qs[3], qs[0]]
            - (All_ang_lat[qs[3], qs[0]] - (All_ang_lat[qs[3] - 1, qs[0]])) / 2
        )
        QS_up_rightx = (
            All_ang_lon[qs[3], qs[1]]
            - (All_ang_lon[qs[3], qs[1]] - (All_ang_lon[qs[3], qs[1] - 1])) / 2
        )
        QS_up_righty = (
            All_ang_lat[qs[3], qs[1]]
            - (All_ang_lat[qs[3], qs[1]] - (All_ang_lat[qs[3] - 1, qs[1]])) / 2
        )
        QS_down_rightx = (
            All_ang_lon[qs[2], qs[1]]
            - (All_ang_lon[qs[2], qs[1]] - (All_ang_lon[qs[2], qs[1] - 1])) / 2
        )
        QS_down_righty = (
            All_ang_lat[qs[2], qs[1]]
            + (All_ang_lat[qs[2], qs[1]] - (All_ang_lat[qs[2] + 1, qs[1]])) / 2
        )
        QS_down_leftx = (
            All_ang_lon[qs[2], qs[0]]
            + (All_ang_lon[qs[2], qs[0]] - (All_ang_lon[qs[2], qs[0] + 1])) / 2
        )
        QS_down_lefty = (
            All_ang_lat[qs[2], qs[0]]
            + (All_ang_lat[qs[2], qs[0]] - (All_ang_lat[qs[2] + 1, qs[0]])) / 2
        )
        # print(qs[3],qs[0],QS_up_leftx)
        ws = window_size
        WS_up_leftx = (
            All_ang_lon[ws[0, 1], ws[1, 0]]
            + (All_ang_lon[ws[0, 1], ws[1, 0]] - (All_ang_lon[ws[0, 1], ws[1, 0] + 1]))
            / 2
        )
        WS_up_lefty = (
            All_ang_lat[ws[0, 1], ws[1, 0]]
            - (All_ang_lat[ws[0, 1], ws[1, 0]] - (All_ang_lat[ws[0, 1] - 1, ws[1, 0]]))
            / 2
        )
        WS_up_rightx = (
            All_ang_lon[ws[0, 1], ws[1, 1]]
            - (All_ang_lon[ws[0, 1], ws[1, 1]] - (All_ang_lon[ws[0, 1], ws[1, 1] - 1]))
            / 2
        )
        WS_up_righty = (
            All_ang_lat[ws[0, 1], ws[1, 1]]
            - (All_ang_lat[ws[0, 1], ws[1, 1]] - (All_ang_lat[ws[0, 1] - 1, ws[1, 1]]))
            / 2
        )
        WS_down_rightx = (
            All_ang_lon[ws[0, 0], ws[1, 1]]
            - (All_ang_lon[ws[0, 0], ws[1, 1]] - (All_ang_lon[ws[0, 0], ws[1, 1] - 1]))
            / 2
        )
        WS_down_righty = (
            All_ang_lat[ws[0, 0], ws[1, 1]]
            + (All_ang_lat[ws[0, 0], ws[1, 1]] - (All_ang_lat[ws[0, 0] + 1, ws[1, 1]]))
            / 2
        )
        WS_down_leftx = (
            All_ang_lon[ws[0, 0], ws[1, 0]]
            + (All_ang_lon[ws[0, 0], ws[1, 0]] - (All_ang_lon[ws[0, 0], ws[1, 0] + 1]))
            / 2
        )
        WS_down_lefty = (
            All_ang_lat[ws[0, 0], ws[1, 0]]
            + (All_ang_lat[ws[0, 0], ws[1, 0]] - (All_ang_lat[ws[0, 0] + 1, ws[1, 0]]))
            / 2
        )

    def sub_q(Q: list) -> list:
        sub_q = []
        i_b1 = 0
        i_b2 = 0
        for i in range(len(Q)):
            if Q[i] == "B":
                i_b2 = i + 1
                sub_q.append([i_b1, i_b2])
                i_b1 = i_b2
        return sub_q

    _q = sub_q(q)
    _c = -1
    _nl = int((len(q) - len(_q)) / 3) + len(_q) + 1
    # conv_c = 0
    mean_pos = []
    plt.rcParams.update({"font.size": 22})
    fig, axis = plt.subplots(_nl, 3, figsize=(24, 8.2 * _nl), constrained_layout=True)

    for i in range(len(q)):
        if q[i] == "I":
            _c += 1
            minmax = {"interval": AsymmetricPercentileInterval(1, 99)}
            if type(max_I) != type(None) and type(min_I) != type(None):
                minmax = {"vmin": min_I, "vmax": max_I}

            norm = ImageNormalize(p[i, t], **minmax, stretch=SqrtStretch())
            if "raster" in kwargs:
                im = axis[_c, 0].pcolormesh(
                    ang_lon, ang_lat, p[i, t], norm=norm, cmap="magma"
                )
            else:
                im = axis[_c, 0].imshow(
                    p[i, t], aspect="auto", origin="lower", norm=norm, cmap="magma"
                )
            axis[_c, 0].set_title(
                "Intensity ($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)"
            )
            axis[_c, 0].set_xlabel("Helioprojective longitude \n (arcsec)")
            axis[_c, 0].set_ylabel("Helioprojective latitude \n (arcsec)")
            axis[_c, 0].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 0], extend=("both" if visualize_saturation else None)
            )

        if q[i] == "x":
            mean_x = np.nanmean(
                p[
                    i,
                    t,
                    quite_sun[2] - ws[0, 0] : quite_sun[3] + ws[0, 0],
                    quite_sun[0] - ws[1, 0] : quite_sun[1] - ws[1, 0],
                ]
            )
            # print(quite_sun)
            # print(ws)
            # print([i,t,quite_sun[2]-ws[0,1],quite_sun[3]+ws[0,1],quite_sun[0]-ws[1,0],quite_sun[1]-ws[1,0]])
            cmap = plt.get_cmap(cmap_doppler).copy()
            (
                cmap.set_extremes(under="yellow", over="green")
                if visualize_saturation
                else 0
            )
            if "raster" in kwargs:
                # print("Calculated mean: ", mean_x)
                im = axis[_c, 1].pcolormesh(
                    ang_lon,
                    ang_lat,
                    (p[i, t] - mean_x) / mean_x * 3 * 10**5,
                    vmin=min_x,
                    vmax=max_x,
                    cmap=cmap,
                )
            else:
                im = axis[_c, 1].imshow(
                    (p[i, t] - mean_x) / mean_x * 3 * 10**5,
                    origin="lower",
                    vmin=min_x,
                    vmax=max_x,
                    aspect="auto",
                    cmap=cmap,
                )
            axis[_c, 1].set_title("Doppler (km/s)")
            axis[_c, 1].set_xlabel("Helioprojective longitude \n (arcsec)")
            axis[_c, 1].set_ylabel("Helioprojective latitude \n (arcsec)")
            axis[_c, 1].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 1], extend=("both" if visualize_saturation else None)
            )
            mean_pos.append(np.nanmean(p[i, t]))
        if q[i] == "s":
            cmap = plt.get_cmap("hot").copy()
            (
                cmap.set_extremes(under="green", over="violet")
                if visualize_saturation
                else 0
            )
            if "raster" in kwargs:
                im = axis[_c, 2].pcolormesh(
                    ang_lon, ang_lat, p[i, t], cmap=cmap, vmax=max_s, vmin=min_s
                )
            else:
                im = axis[_c, 2].imshow(
                    p[i, t], aspect="auto", origin="lower", cmap=cmap
                )

            axis[_c, 2].set_title("$\sigma$ ($\AA$)")
            axis[_c, 2].set_xlabel("Helioprojective longitude \n (arcsec)")
            axis[_c, 2].set_ylabel("Helioprojective latitude \n (arcsec)")
            axis[_c, 2].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 2], extend=("both" if visualize_saturation else None)
            )
        if q[i] == "B":
            _c += 1
            minmax = (
                {"interval": AsymmetricPercentileInterval(1, 99)}
                if type(max_B) == type(None)
                else {"vmin": min_B, "vmax": max_B}
            )
            norm = ImageNormalize(p[i, t], **minmax, stretch=SqrtStretch())

            if "raster" in kwargs:
                im = axis[_c, 0].pcolormesh(
                    ang_lon, ang_lat, p[i, t], norm=norm, cmap="magma"
                )
            else:
                im = axis[_c, 0].imshow(
                    p[i, t], aspect="auto", origin="lower", norm=norm, cmap="magma"
                )
            axis[_c, 0].set_title(
                "Background Intensity\n($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)"
            )
            axis[_c, 0].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 0], extend=("both" if visualize_saturation else None)
            )
            axis[_c, 0].set_xlabel("Helioprojective longitude \n (arcsec)")
            axis[_c, 0].set_ylabel("Helioprojective latitude \n (arcsec)")
            axis[_c, 2].remove()
            if type(convlist) == type(None):
                axis[_c, 1].remove()
            else:
                data_conv = convlist[conv_c, t, :, :].copy()
                # designing colormap -> color choice and their number
                vals = (np.unique(data_conv)).astype(np.int8)
                print(vals)
                # print("vals",vals)
                N = len(vals)
                # print(int(N**(1/3)),N**(1/3),'int(N**(1/3)),N**(1/3)')
                n = int(N ** (1 / 3)) + (0 if int(N ** (1 / 3)) == N ** (1 / 3) else 1)
                yn = N ** (1 / 3)
                # print("N,n",N,n)
                col_dict = {}
                i = 0
                for b in range(n):
                    if i == N:
                        break
                    for g in range(n):
                        if i == N:
                            break
                        for r in range(n):
                            if i == N:
                                break
                            col_dict[vals[i]] = (
                                1 - r / max(n - 1, 1),
                                1 - g / max(n - 1, 1),
                                1 - b / max(n - 1, 1),
                            )
                            i += 1
                # print("col_dict",col_dict)
                # designing colormap -> constructing the cmap
                cmap = ListedColormap([col_dict[x] for x in col_dict.keys()])

                # In case you want the labels to show other than the numbers put the list of these labels here
                labels = vals.copy()
                len_lab = len(labels)

                # prepare for the normalizer
                norm_bins = np.sort([*col_dict.keys()]) + 0.5
                # print("sort",norm_bins)
                norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
                # print("insert", norm_bins)
                # Make normalizer and formatter
                norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
                fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

                if "raster" in kwargs:
                    im = axis[_c, 1].pcolormesh(
                        ang_lon, ang_lat, data_conv, cmap=cmap, norm=norm
                    )
                else:
                    im = axis[_c, 1].imshow(
                        data_conv,
                        aspect="auto",
                        origin="lower",
                        cmap=cmap,
                        norm=norm,
                    )
                axis[_c, 1].set_xlabel("Helioprojective longitude \n (arcsec)")
                axis[_c, 1].set_ylabel("Helioprojective latitude \n (arcsec)")
                axis[_c, 1].set_title("convolution level map")
                axis[_c, 1].set_aspect("equal")

                diff = norm_bins[1:] - norm_bins[:-1]
                tickz = norm_bins[:-1] + diff / 2
                plt.colorbar(im, ax=axis[_c, 1], ticks=tickz)
                # conv_c  += 1

    # plt.rcParams.update({'font.size': 16})
    if "raster" in kwargs:

        mn = np.nanmean(
            w[:, :, :, :], axis=(0, 1)  # ws[0,0]:ws[0,1],  # ws[1,0]:ws[1,1]
        )
        minmax = {"interval": AsymmetricPercentileInterval(1, 99)}
        if type(max_I) != type(None) and type(min_I) != type(None):
            minmax = {"vmin": min_I, "vmax": max_I}

        norm = ImageNormalize(mn, **minmax, stretch=SqrtStretch())
        im = axis[-1, 0].pcolormesh(
            All_ang_lon, All_ang_lat, mn, norm=norm, cmap="magma"
        )
        # axis[-1,0].set_aspect('auto')
        plt.colorbar(
            im, ax=axis[-1, 0], extend=("both" if visualize_saturation else None)
        )

        axis[-1, 0].plot(
            [QS_up_leftx, QS_up_rightx, QS_down_rightx, QS_down_leftx, QS_up_leftx],
            [QS_up_lefty, QS_up_righty, QS_down_righty, QS_down_lefty, QS_up_lefty],
            "o-",
            alpha=1,
            color=(0, 1, 0),
            lw=5,
            label="selected\nquite region",
        )
        axis[-1, 0].plot(
            [WS_up_leftx, WS_up_rightx, WS_down_rightx, WS_down_leftx, WS_up_leftx],
            [WS_up_lefty, WS_up_righty, WS_down_righty, WS_down_lefty, WS_up_lefty],
            "o-",
            alpha=1,
            color=(1, 0, 0),
            lw=3,
            label="Analysed region",
        )

    else:
        mn = np.nanmean(w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 1))
        norm = ImageNormalize(
            mn, AsymmetricPercentileInterval(1, 99), stretch=SqrtStretch()
        )

        im = axis[-1, 0].imshow(mn, norm=norm, cmap="magma", origin="lower")
        axis[-1, 0].set_xlabel("Helioprojective longitude \n (arcsec)")
        axis[-1, 0].set_ylabel("Helioprojective latitude \n (arcsec)")

        lims = axis[-1, 0].get_xlim(), axis[-1, 0].get_ylim()
        axis[-1, 0].set_aspect("auto")
        plt.colorbar(
            im, ax=axis[-1, 0], extend=("both" if visualize_saturation else None)
        )

        axis[-1, 0].axvspan(
            xmin=quite_sun[0],
            xmax=quite_sun[1],
            ymin=quite_sun[2] / (axis[-1, 0].get_ylim()[1] - axis[-1, 0].get_ylim()[0]),
            ymax=quite_sun[3] / (axis[-1, 0].get_ylim()[1] - axis[-1, 0].get_ylim()[0]),
            alpha=0.5,
            lw=2,
            color=(0, 1, 0),
            label="selected\nquite region",
        )

    axis[-1, 0].set_title("original averaged\n over spectrum")
    axis[-1, 0].set_xlabel("Helioprojective longitude \n (arcsec)")
    axis[-1, 0].set_ylabel("Helioprojective latitude \n (arcsec)")
    axis[-1, 1].step(
        sa,
        np.nanmean(w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 2, 3)),
    )

    # print(mean_pos)
    # print(sa)
    for x in mean_pos:
        axis[-1, 1].axvline(x, ls=":", label="line: {:02d}".format(mean_pos.index(x)))
    axis[-1, 1].legend()
    axis[-1, 1].set_title(
        "original average spectrum/\naverage line positions/\n segments"
    )

    axis[-1, 2].remove()
    fig.suptitle(suptitle + "\n" + raster.meta["DATE_SUN"][:-4], fontsize=20)
    if type(segmentation) != type(None):
        if len(segmentation.shape) != 1:
            for seg in segmentation:
                color = np.random.rand(3)
                color = 0.8 * color / np.sqrt(np.sum(color**2))
                axis[-1, 1].axvspan(seg[0], seg[1], alpha=0.5, color=color)
        else:
            seg = segmentation
            color = np.random.rand(3)
            color = 0.8 * color / np.sqrt(np.sum(color**2))
            axis[-1, 1].axvspan(seg[0], seg[1], alpha=0.5, color=color)
    for ax in axis.flatten():
        ax.ticklabel_format(useOffset=False)
        ax.ticklabel_format(useOffset=False)
    # fig.tight_layout()
    if save:
        plt.savefig(filename)

    return fig, axis, All_ang_lon, All_ang_lat, ang_lon, ang_lat


def plot_summary_Miho_1line(
    spectrum_axis: np.array,
    window: np.ndarray,
    paramlist: np.ndarray,
    quentity: list,
    convlist=None,
    suptitle="",
    window_size: np.ndarray = np.array([[0, -1], [0, -1]]),
    t: int = 0,
    segmentation=None,
    save=True,
    filename="./imgs/res.pdf",
    quite_sun: np.ndarray = np.array([0, -1, 0, -1]),
    min_I=None,
    max_I=None,
    min_x=-50,
    max_x=+50,
    min_s=None,
    max_s=None,
    min_B=None,
    max_B=None,
    visualize_saturation=False,
    **kwargs
):
    if True:
        w = window
        q = quentity
        p = paramlist
        ws = window_size
        sa = spectrum_axis
        cmap_doppler = "twilight_shifted"

        raster = kwargs["raster"]
        # ang_lat = raster.celestial.data[ws[0,0]:ws[0,1],ws[1,0]:ws[1,1]].lat.deg
        # ang_lon = raster.celestial.data[ws[0,0]:ws[0,1],ws[1,0]:ws[1,1]].lon.deg
        ang_lat = raster.celestial.data[:, :].lat.deg
        ang_lon = raster.celestial.data[:, :].lon.deg
        All_ang_lat = raster.celestial.data.lat.deg
        All_ang_lon = raster.celestial.data.lon.deg

        ang_lat2 = ang_lat.copy()
        ang_lon2 = ang_lon.copy()
        All_ang_lat2 = All_ang_lat.copy()
        All_ang_lon2 = All_ang_lon.copy()

        All_ang_lon2[All_ang_lon <= 180] = All_ang_lon[All_ang_lon <= 180] * 3600
        All_ang_lon2[All_ang_lon > 180] = (All_ang_lon[All_ang_lon > 180] - 360) * 3600

        ang_lon2[ang_lon <= 180] = ang_lon[ang_lon <= 180] * 3600
        ang_lon2[ang_lon > 180] = (ang_lon[ang_lon > 180] - 360) * 3600

        All_ang_lat2[All_ang_lat <= 180] = All_ang_lat[All_ang_lat <= 180] * 3600
        All_ang_lat2[All_ang_lat > 180] = (All_ang_lat[All_ang_lat > 180]) * 3600

        ang_lat2[ang_lat <= 180] = ang_lat[ang_lat <= 180] * 3600
        ang_lat2[ang_lat > 180] = (ang_lat[ang_lat > 180]) * 3600
        ang_lat = ang_lat2.copy()
        ang_lon = ang_lon2.copy()
        All_ang_lat = All_ang_lat2.copy()
        All_ang_lon = All_ang_lon2.copy()

        if type(quite_sun) == np.ndarray:
            qs2 = quite_sun.copy()

            quite_sun = quite_sun.copy()
            # quite_sun[0] = quite_sun[0] - window_size[1,0]
            # quite_sun[2] = quite_sun[2] - window_size[0,0]

            if quite_sun[1] == -1:
                if window_size[1, 1] == -1:
                    quite_sun[1] = window.shape[3]  # - window_size[1,0]
                else:
                    quite_sun[1] = window_size[1, 1]  # - window_size[1,0]
            else:
                pass  # quite_sun[1] = quite_sun[1] - window_size[1,0]
            if quite_sun[3] == -1:
                if window_size[0, 1] == -1:
                    quite_sun[3] = window.shape[2]  # - window_size[0,0]
                else:
                    quite_sun[3] = window_size[0, 1]  # - window_size[0,0]
            else:
                quite_sun[3] = quite_sun[3]  # - window_size[0,0]
            qs = qs2.copy()
            QS_up_leftx = (
                All_ang_lon[qs[3], qs[0]]
                + (All_ang_lon[qs[3], qs[0]] - (All_ang_lon[qs[3], qs[0] + 1])) / 2
            )
            QS_up_lefty = (
                All_ang_lat[qs[3], qs[0]]
                - (All_ang_lat[qs[3], qs[0]] - (All_ang_lat[qs[3] - 1, qs[0]])) / 2
            )
            QS_up_rightx = (
                All_ang_lon[qs[3], qs[1]]
                - (All_ang_lon[qs[3], qs[1]] - (All_ang_lon[qs[3], qs[1] - 1])) / 2
            )
            QS_up_righty = (
                All_ang_lat[qs[3], qs[1]]
                - (All_ang_lat[qs[3], qs[1]] - (All_ang_lat[qs[3] - 1, qs[1]])) / 2
            )
            QS_down_rightx = (
                All_ang_lon[qs[2], qs[1]]
                - (All_ang_lon[qs[2], qs[1]] - (All_ang_lon[qs[2], qs[1] - 1])) / 2
            )
            QS_down_righty = (
                All_ang_lat[qs[2], qs[1]]
                + (All_ang_lat[qs[2], qs[1]] - (All_ang_lat[qs[2] + 1, qs[1]])) / 2
            )
            QS_down_leftx = (
                All_ang_lon[qs[2], qs[0]]
                + (All_ang_lon[qs[2], qs[0]] - (All_ang_lon[qs[2], qs[0] + 1])) / 2
            )
            QS_down_lefty = (
                All_ang_lat[qs[2], qs[0]]
                + (All_ang_lat[qs[2], qs[0]] - (All_ang_lat[qs[2] + 1, qs[0]])) / 2
            )
            # print(qs[3],qs[0],QS_up_leftx)
            ws = window_size
            WS_up_leftx = (
                All_ang_lon[ws[0, 1], ws[1, 0]]
                + (
                    All_ang_lon[ws[0, 1], ws[1, 0]]
                    - (All_ang_lon[ws[0, 1], ws[1, 0] + 1])
                )
                / 2
            )
            WS_up_lefty = (
                All_ang_lat[ws[0, 1], ws[1, 0]]
                - (
                    All_ang_lat[ws[0, 1], ws[1, 0]]
                    - (All_ang_lat[ws[0, 1] - 1, ws[1, 0]])
                )
                / 2
            )
            WS_up_rightx = (
                All_ang_lon[ws[0, 1], ws[1, 1]]
                - (
                    All_ang_lon[ws[0, 1], ws[1, 1]]
                    - (All_ang_lon[ws[0, 1], ws[1, 1] - 1])
                )
                / 2
            )
            WS_up_righty = (
                All_ang_lat[ws[0, 1], ws[1, 1]]
                - (
                    All_ang_lat[ws[0, 1], ws[1, 1]]
                    - (All_ang_lat[ws[0, 1] - 1, ws[1, 1]])
                )
                / 2
            )
            WS_down_rightx = (
                All_ang_lon[ws[0, 0], ws[1, 1]]
                - (
                    All_ang_lon[ws[0, 0], ws[1, 1]]
                    - (All_ang_lon[ws[0, 0], ws[1, 1] - 1])
                )
                / 2
            )
            WS_down_righty = (
                All_ang_lat[ws[0, 0], ws[1, 1]]
                + (
                    All_ang_lat[ws[0, 0], ws[1, 1]]
                    - (All_ang_lat[ws[0, 0] + 1, ws[1, 1]])
                )
                / 2
            )
            WS_down_leftx = (
                All_ang_lon[ws[0, 0], ws[1, 0]]
                + (
                    All_ang_lon[ws[0, 0], ws[1, 0]]
                    - (All_ang_lon[ws[0, 0], ws[1, 0] + 1])
                )
                / 2
            )
            WS_down_lefty = (
                All_ang_lat[ws[0, 0], ws[1, 0]]
                + (
                    All_ang_lat[ws[0, 0], ws[1, 0]]
                    - (All_ang_lat[ws[0, 0] + 1, ws[1, 0]])
                )
                / 2
            )
        elif type(quite_sun) == float:
            pass
        else:
            raise Exception("the algorithm accepts only np.ndarray or floats")

        def sub_q(Q: list) -> list:
            sub_q = []
            i_b1 = 0
            i_b2 = 0
            for i in range(len(Q)):
                if Q[i] == "B":
                    i_b2 = i + 1
                    sub_q.append([i_b1, i_b2])
                    i_b1 = i_b2
            return sub_q

        mean_pos = []
        _q = sub_q(q)
        _c = -1
        _nl = int((len(q) - len(_q)) / 3) + len(_q) + 1
        # conv_c = 0
        mpl.style.use("classic")
        plt.rcParams.update(
            {
                "font.size": 30,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "axes.facecolor": "white",
                "text.color": "black",
                "axes.labelcolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
            }
        )
        fig = plt.figure(figsize=(24, 24))
        gs0 = fig.add_gridspec(2, 2)
        gs1 = gs0[0, 0].subgridspec(2, 1)
        gs2 = gs1[1].subgridspec(1, 2)

        # fig, axis = plt.subplots(2,2, figsize=(24,8.2*_nl))

        for i in range(len(q)):
            if q[i] == "I":
                _c += 1
                if _c > 0:
                    raise Exception(
                        "you called the function plot_summary_Miho_1line it is for ONE line not for multiple lines DUMBASS!!! "
                    )
                axI = fig.add_subplot(gs0[0, 1])
                minmax = {"interval": AsymmetricPercentileInterval(1, 99)}
                if type(max_I) != type(None) and type(min_I) != type(None):
                    minmax = {"vmin": min_I, "vmax": max_I}

                norm = ImageNormalize(p[i, t], **minmax, stretch=SqrtStretch())
                im = axI.pcolormesh(
                    ang_lon, ang_lat, p[i, t], norm=norm, cmap="magma", rasterized=True
                )
                axI.set_title("Intensity ($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)")
                ymin = np.nanmin([ang_lat[np.logical_not(np.isnan(p[i, t]))]])
                ymax = np.nanmax([ang_lat[np.logical_not(np.isnan(p[i, t]))]])
                ylim = [ymin, ymax]
                xmin = np.nanmin([ang_lon[np.logical_not(np.isnan(p[i, t]))]])
                xmax = np.nanmax([ang_lon[np.logical_not(np.isnan(p[i, t]))]])
                xlim = [xmin, xmax]

                axI.set_xlabel("Helioprojective longitude \n (arcsec)")
                axI.set_ylabel("Helioprojective latitude \n (arcsec)")
                axI.set_ylim(ylim)
                axI.set_xlim(xlim)
                axI.set_aspect("equal")
                axI.xaxis.set_tick_params(rotation=45)

                cb = plt.colorbar(
                    im,
                    shrink=0.75,
                    ax=axI,
                    extend=("both" if visualize_saturation else None),
                )
                cb.solids.set_rasterized(True)
            if q[i] == "x":
                axx = fig.add_subplot(gs0[1, 0])
                if type(quite_sun) == np.ndarray:
                    mean_x = np.nanmean(
                        p[
                            i,
                            t,
                            quite_sun[2] : quite_sun[3],
                            quite_sun[0] : quite_sun[1],
                        ]
                    )
                else:
                    mean_x = quite_sun
                mean_pos.append(mean_x)
                cmap = plt.get_cmap(cmap_doppler).copy()
                (
                    cmap.set_extremes(under="yellow", over="green")
                    if visualize_saturation
                    else 0
                )

                # print("Calculated mean: ", mean_x)
                im = axx.pcolormesh(
                    ang_lon,
                    ang_lat,
                    (p[i, t] - mean_x) / mean_x * 3 * 10**5,
                    vmin=min_x,
                    vmax=max_x,
                    cmap=cmap,
                    rasterized=True,
                )

                axx.set_title("Doppler (km/s)")
                axx.set_xlabel("Helioprojective longitude \n (arcsec)")
                axx.set_ylabel("Helioprojective latitude \n (arcsec)")
                axx.set_ylim(ylim)
                axx.set_xlim(xlim)

                axx.set_aspect("equal")
                axx.xaxis.set_tick_params(rotation=45)

                cb = plt.colorbar(
                    im,
                    shrink=0.75,
                    ax=axx,
                    extend=("both" if visualize_saturation else None),
                )
                cb.solids.set_rasterized(True)
            if q[i] == "s":
                axs = fig.add_subplot(gs0[1, 1])
                cmap = plt.get_cmap("hot").copy()
                (
                    cmap.set_extremes(under="green", over="violet")
                    if visualize_saturation
                    else 0
                )
                im = axs.pcolormesh(
                    ang_lon,
                    ang_lat,
                    p[i, t],
                    cmap=cmap,
                    vmax=max_s,
                    vmin=min_s,
                    rasterized=True,
                )

                axs.set_title("$\sigma$ ($\AA$)")
                axs.set_xlabel("Helioprojective longitude \n (arcsec)")
                axs.set_ylabel("Helioprojective latitude \n (arcsec)")
                axs.set_ylim(ylim)
                axs.set_xlim(xlim)

                axs.set_aspect("equal")
                axs.xaxis.set_tick_params(rotation=45)

                cb = plt.colorbar(
                    im,
                    shrink=0.75,
                    ax=axs,
                    extend=("both" if visualize_saturation else None),
                )
                cb.solids.set_rasterized(True)

        mn = np.nanmean(
            w[:, :, :, :], axis=(0, 1)  # ws[0,0]:ws[0,1],  # ws[1,0]:ws[1,1]
        )
        axAI = fig.add_subplot(gs2[0])
        minmax = {"interval": AsymmetricPercentileInterval(1, 99)}
        if type(max_I) != type(None) and type(min_I) != type(None):
            minmax = {"vmin": min_I, "vmax": max_I}

        norm = ImageNormalize(mn, **minmax, stretch=SqrtStretch())
        im = axAI.pcolormesh(
            All_ang_lon, All_ang_lat, mn, norm=norm, cmap="magma", rasterized=True
        )
        # axAI.set_aspect('auto')
        axAI.set_ylim(ylim)
        axAI.set_xlim(xlim)

        cb = plt.colorbar(
            im, shrink=0.75, ax=axAI, extend=("both" if visualize_saturation else None)
        )
        cb.solids.set_rasterized(True)
        cb.ax.tick_params(labelsize=16)

        axAI.plot(
            [QS_up_leftx, QS_up_rightx, QS_down_rightx, QS_down_leftx, QS_up_leftx],
            [QS_up_lefty, QS_up_righty, QS_down_righty, QS_down_lefty, QS_up_lefty],
            "o-",
            alpha=1,
            color=(0, 1, 0),
            lw=5,
            label="Selected\nquite region",
        )
        axAI.plot(
            [WS_up_leftx, WS_up_rightx, WS_down_rightx, WS_down_leftx, WS_up_leftx],
            [WS_up_lefty, WS_up_righty, WS_down_righty, WS_down_lefty, WS_up_lefty],
            "o-",
            alpha=1,
            color=(1, 0, 0),
            lw=3,
            label="Analysed region",
        )

        axAI.set_title("Average intensity", fontsize=16)
        axAI.set_xlabel("Helioprojective longitude \n (arcsec)", fontsize=16)
        axAI.set_ylabel("Helioprojective latitude \n (arcsec)", fontsize=16)
        axAI.set_aspect("equal")
        axAI.xaxis.set_tick_params(labelsize=16)
        axAI.yaxis.set_tick_params(labelsize=16)
        axAI.xaxis.set_tick_params(rotation=45)

        axAS = fig.add_subplot(gs2[1])
        axAS.step(
            sa,
            np.nanmean(
                w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 2, 3)
            ),
        )

        # print(mean_pos)
        # print(sa)
        for x in mean_pos:
            axAS.axvline(x, ls=":", label="line: {:02d}".format(mean_pos.index(x)))
        axAS.legend(fontsize=16)
        axAS.set_title("Average spectrum", fontsize=16)
        axAS.set_ylabel("Intensity", fontsize=16)
        axAS.set_xlabel("Wavelength $(\AA)$", fontsize=16)
        axAS.xaxis.set_tick_params(labelsize=16)
        axAS.xaxis.set_tick_params(labelsize=16)
        t = axAS.yaxis.get_offset_text()
        t.set_size(16)
        t = axAS.xaxis.get_offset_text()
        t.set_size(16)

        fig.text(
            0.08, 0.80, suptitle + "\n" + raster.meta["DATE_SUN"][:-4], fontsize=60
        )
        if type(segmentation) != type(None):
            if len(segmentation.shape) != 1:
                for seg in segmentation:
                    color = np.random.rand(3)
                    color = 0.8 * color / np.sqrt(np.sum(color**2))
                    axAS.axvspan(seg[0], seg[1], alpha=0.5, color=color)
            else:
                seg = segmentation
                color = np.random.rand(3)
                color = 0.8 * color / np.sqrt(np.sum(color**2))
                axAS.axvspan(seg[0], seg[1], alpha=0.5, color=color)
        plt.tight_layout()
        if save:
            plt.savefig(filename)

        return (
            fig,
            [axI, axx, axs, axAI, axAS],
            All_ang_lon,
            All_ang_lat,
            ang_lon,
            ang_lat,
            quite_sun,
        )


def plot_window(
    spectrum_axis: np.array,
    window: np.ndarray,
    paramlist: np.ndarray,
    quentity: list,
    convlist=None,
    suptitle="",
    window_size: np.ndarray = np.array([[0, -1], [0, -1]]),
    t: int = 0,
    segmentation=None,
    save=False,
    filename="./imgs/res.jpg",
    quite_sun: np.ndarray = np.array([0, -1, 0, -1]),
    min_vel=-100,
    max_vel=+100,
):
    cmap_doppler = "twilight_shifted"
    quite_sun = quite_sun.copy()
    quite_sun[0] = quite_sun[0] - window_size[1, 0]
    quite_sun[2] = quite_sun[2] - window_size[0, 0]

    if quite_sun[1] == -1:
        if window_size[1, 1] == -1:
            quite_sun[1] = window.shape[3] - window_size[1, 0]
        else:
            quite_sun[1] = window_size[1, 1] - window_size[1, 0]
    else:
        quite_sun[1] = quite_sun[1] - window_size[1, 0]
    if quite_sun[3] == -1:
        if window_size[0, 1] == -1:
            quite_sun[3] = window.shape[2] - window_size[0, 0]
        else:
            quite_sun[3] = window_size[0, 1] - window_size[0, 0]
    else:
        quite_sun[3] = quite_sun[3] - window_size[0, 0]

    w = window
    q = quentity
    p = paramlist
    ws = window_size
    sa = spectrum_axis

    def sub_q(Q: list) -> list:
        sub_q = []
        i_b1 = 0
        i_b2 = 0
        for i in range(len(Q)):
            if Q[i] == "B":
                i_b2 = i + 1
                sub_q.append([i_b1, i_b2])
                i_b1 = i_b2
        return sub_q

    _q = sub_q(q)
    _c = -1
    _nl = int((len(q) - len(_q)) / 3) + len(_q) + 1
    # conv_c = 0
    mean_pos = []
    plt.rcParams.update({"font.size": 22})
    fig, axis = plt.subplots(_nl, 3, figsize=(24, 6 * _nl))

    for i in range(len(q)):
        if q[i] == "I":
            _c += 1
            norm = ImageNormalize(
                p[i, t], AsymmetricPercentileInterval(1, 99), stretch=SqrtStretch()
            )
            im = axis[_c, 0].imshow(
                p[i, t], aspect="auto", origin="lower", norm=norm, cmap="magma"
            )
            axis[_c, 0].set_title(
                "Intensity ($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)"
            )
            plt.colorbar(
                im, ax=axis[_c, 0], extend=("both" if visualize_saturation else None)
            )

        if q[i] == "x":
            mean_x = np.nanmean(
                p[i, t, quite_sun[0] : quite_sun[1], quite_sun[2] : quite_sun[3]]
            )
            im = axis[_c, 1].imshow(
                (p[i, t] - mean_x) / mean_x * 3 * 10**5,
                origin="lower",
                vmin=min_vel,
                vmax=max_vel,
                aspect="auto",
                cmap=cmap_doppler,
            )
            axis[_c, 1].set_title("Doppler (km/s)")
            plt.colorbar(
                im, ax=axis[_c, 1], extend=("both" if visualize_saturation else None)
            )
            mean_pos.append(np.nanmean(p[i, t]))

        if q[i] == "s":
            im = axis[_c, 2].imshow(p[i, t], aspect="auto", origin="lower", cmap="hot")
            axis[_c, 2].set_title("$\sigma$ ($\AA$)")
            plt.colorbar(
                im, ax=axis[_c, 2], extend=("both" if visualize_saturation else None)
            )

        if q[i] == "B":
            _c += 1
            norm = ImageNormalize(
                p[i, t], AsymmetricPercentileInterval(1, 99), stretch=SqrtStretch()
            )
            im = axis[_c, 0].imshow(
                p[i, t], aspect="auto", origin="lower", norm=norm, cmap="magma"
            )
            axis[_c, 0].set_title(
                "Background Intensity ($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)"
            )
            plt.colorbar(
                im, ax=axis[_c, 0], extend=("both" if visualize_saturation else None)
            )

            axis[_c, 2].remove()
            if type(convlist) == type(None):
                axis[_c, 1].remove()
            else:
                im = axis[_c, 1].imshow(
                    convlist[conv_c, t, :, :],
                    aspect="auto",
                    origin="lower",
                    cmap="Dark2",
                    vmin=-0.5,
                    vmax=7.5,
                )
                axis[_c, 1].set_title("convolution level map")
                plt.colorbar(
                    im,
                    ax=axis[_c, 1],
                    extend=("both" if visualize_saturation else None),
                )
                # conv_c  += 1

    mn = np.nanmean(w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 1))
    norm = ImageNormalize(
        mn, AsymmetricPercentileInterval(1, 99), stretch=SqrtStretch()
    )
    # print(axis.shape)

    im = axis[-1, 0].imshow(mn, aspect="auto", norm=norm, cmap="magma", origin="lower")
    plt.colorbar(im, ax=axis[-1, 0], extend=("both" if visualize_saturation else None))

    axis[-1, 0].axvspan(
        xmin=quite_sun[0],
        xmax=quite_sun[1],
        ymin=quite_sun[2] / (axis[-1, 0].get_ylim()[1] - axis[-1, 0].get_ylim()[0]),
        ymax=quite_sun[3] / (axis[-1, 0].get_ylim()[1] - axis[-1, 0].get_ylim()[0]),
        alpha=0.5,
        color=(0, 1, 0),
        label="selected\nquite region",
    )
    axis[-1, 0].set_title("original averaged\n over spectrum")

    axis[-1, 1].plot(
        sa,
        np.nanmean(w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 2, 3)),
    )

    # print(mean_pos)
    # print(sa)
    for x in mean_pos:
        axis[-1, 1].axvline(x, ls=":", label="line: {:02d}".format(mean_pos.index(x)))
    axis[-1, 1].legend()
    axis[-1, 1].set_title(
        "original average spectrum/ average line positions/\n segments"
    )
    axis[-1, 2].remove()
    fig.suptitle(suptitle, fontsize=20)
    if type(segmentation) != type(None):
        if len(segmentation.shape) != 1:
            for seg in segmentation:
                color = np.random.rand(3)
                color = 0.8 * color / np.sqrt(np.sum(color**2))
                axis[-1, 1].axvspan(seg[0], seg[1], alpha=0.5, color=color)
        else:
            seg = segmentation
            color = np.random.rand(3)
            color = 0.8 * color / np.sqrt(np.sum(color**2))
            axis[-1, 1].axvspan(seg[0], seg[1], alpha=0.5, color=color)
    fig.tight_layout()
    if save:
        plt.savefig(filename)

    return fig, axis


def plot_error(
    covlist,
    paramlist,
    quentity,
    fig=None,
    axis=None,
    t: int = 0,
    alpha=1,
    label="",
    save=True,
    filename="./imgs/res.jpg",
    min_hist=-2,
    max_hist=1,
):
    cl = covlist
    q = quentity
    p = paramlist

    def sub_q(Q: list) -> list:
        sub_q = []
        i_b1 = 0
        i_b2 = 0
        for i in range(len(Q)):
            if Q[i] == "B":
                i_b2 = i + 1
                sub_q.append([i_b1, i_b2])
                i_b1 = i_b2
        return np.array(sub_q)

    _q = sub_q(q)
    _nl = int((len(q) - len(_q)) / 3) + len(_q)
    _c = -1
    if type(axis) == type(None):
        fig, axis = plt.subplots(_nl, 3, figsize=(24, 6 * _nl))
    if False:
        fig2, axis2 = plt.subplots(5, 5, figsize=(12, 3 * 5))
        axis2 = axis2.flatten()
    i = 0
    for i in range(len(_q)):
        for j in range(_q[i, 0], _q[i, 1]):
            _j = j - _q[i, 0]
            q_val = q[j]
            data = cl[j, j, t]
            _data = p[j, t]
            if q_val == "I":
                _c += 1
                axis[_c, 0].hist(
                    (np.sqrt(data) / _data).flatten(),
                    bins=10 ** np.arange(min_hist - 1, max_hist - 1, 0.1),
                    log=True,
                    alpha=alpha,
                    label=label,
                )
                axis[_c, 0].set_xscale("log")
                axis[_c, 0].set_yscale("log")
                axis[_c, 0].set_title(r"${\Delta I}{ I}$")
            if q_val == "x":
                axis[_c, 1].hist(
                    (np.sqrt(data) / _data).flatten(),
                    bins=10 ** np.arange(min_hist - 4, max_hist - 4, 0.1),
                    log=True,
                    alpha=alpha,
                    label=label,
                )
                axis[_c, 1].set_xscale("log")
                axis[_c, 1].set_yscale("log")
                axis[_c, 1].set_title(r"${\Delta \lambda_0}{\lambda_0}$")
            if q_val == "s":
                axis[_c, 2].hist(
                    (np.sqrt(data) / _data).flatten(),
                    bins=10 ** np.arange(min_hist - 1, max_hist - 1, 0.1),
                    log=True,
                    alpha=alpha,
                    label=label,
                )
                axis[_c, 2].set_xscale("log")
                axis[_c, 2].set_yscale("log")
                axis[_c, 2].set_title(r"${\Delta \sigma}{\sigma}$")
            if q_val == "B":
                _c += 1
                axis[_c, 0].hist(
                    (np.sqrt(data) / _data).flatten(),
                    bins=10 ** np.arange(min_hist, max_hist, 0.1),
                    log=True,
                    alpha=alpha,
                    label=label,
                )
                axis[_c, 0].set_xscale("log")
                axis[_c, 0].set_yscale("log")
                axis[_c, 0].set_title(r"${\Delta B}{ B}$")
                try:
                    axis[_c, 1].remove()
                    axis[_c, 2].remove()
                except:
                    pass
    fig.tight_layout()
    if save:
        plt.savefig(filename)
    return fig, axis


def plot_sumup(
    raster: np.ndarray,
    kwOrOrder: str or int,
    paramlist: np.ndarray,
    quentity: list,
    convlist: np.ndarray = None,
    locklist: np.ndarray = None,
    lockline: int = 0,
    suptitle: str = None,
    window_size: np.ndarray = np.array([[0, -1], [0, -1]]),
    t: int = 0,  # what is this
    segmentation=None,
    save=False,
    filename="./imgs/res.jpg",
    linenames=None,
    quite_sun: np.ndarray = np.array([0, -1, 0, -1]),
    maxwell_correction: bool = True,
    min_I=None,
    max_I=None,
    min_V=-80,
    max_V=+80,
    min_W=None,
    max_W=None,
    min_B=None,
    max_B=None,
    visualize_saturation=True,
    verbose=1,
):

    unq = spu.unique_windows(raster)
    if type(kwOrOrder) == str:
        kw = kwOrOrder
    else:
        kw = unq[kwOrOrder]
    if True:  # Setting The parametrers and the variables
        r = raster
        w = r[kw].data
        q = quentity
        p = paramlist
        ws = window_size
        sa = r[kw].spectral_axis * 10**10
        qs = quite_sun
        cmap_doppler = "twilight_shifted"
        ratio = 4
        title_size = 4 * ratio
        label_size = 2 * ratio
        sup_title_size = 6 * ratio
    if True:  # Coordinate matrix
        All_ang_lat = r[kw].celestial.data.lat.arcsec
        All_ang_lon = r[kw].celestial.data.lon.arcsec

        All_ang_lon[All_ang_lon >= 180 * 3600] = (
            All_ang_lon[All_ang_lon >= 180 * 3600] - 360 * 3600
        )
        All_ang_lat[All_ang_lat >= 180 * 3600] = (
            All_ang_lat[All_ang_lat >= 180 * 3600] - 360 * 3600
        )
    if True:  # Adjusting quite sun
        if qs[1] < 0:
            if ws[1, 1] < 0:
                qs[1] = w.shape[3] + 2 + ws[1, 1] + qs[1]
            else:
                qs[1] = ws[1, 1] + 1 + qs[1]
        if qs[3] == -1:
            if ws[0, 1] < 0:
                qs[3] = w.shape[2] + 2 + ws[0, 1] + qs[3]
            else:
                qs[3] = ws[0, 1] + 1 + qs[3]
    if True:  # quite sun and window size coordinates
        QS_up_leftx = (
            All_ang_lon[qs[3], qs[0]]
            + (All_ang_lon[qs[3], qs[0]] - (All_ang_lon[qs[3], qs[0] + 1])) / 2
        )
        QS_up_lefty = (
            All_ang_lat[qs[3], qs[0]]
            - (All_ang_lat[qs[3], qs[0]] - (All_ang_lat[qs[3] - 1, qs[0]])) / 2
        )
        QS_up_rightx = (
            All_ang_lon[qs[3], qs[1]]
            - (All_ang_lon[qs[3], qs[1]] - (All_ang_lon[qs[3], qs[1] - 1])) / 2
        )
        QS_up_righty = (
            All_ang_lat[qs[3], qs[1]]
            - (All_ang_lat[qs[3], qs[1]] - (All_ang_lat[qs[3] - 1, qs[1]])) / 2
        )
        QS_down_rightx = (
            All_ang_lon[qs[2], qs[1]]
            - (All_ang_lon[qs[2], qs[1]] - (All_ang_lon[qs[2], qs[1] - 1])) / 2
        )
        QS_down_righty = (
            All_ang_lat[qs[2], qs[1]]
            + (All_ang_lat[qs[2], qs[1]] - (All_ang_lat[qs[2] + 1, qs[1]])) / 2
        )
        QS_down_leftx = (
            All_ang_lon[qs[2], qs[0]]
            + (All_ang_lon[qs[2], qs[0]] - (All_ang_lon[qs[2], qs[0] + 1])) / 2
        )
        QS_down_lefty = (
            All_ang_lat[qs[2], qs[0]]
            + (All_ang_lat[qs[2], qs[0]] - (All_ang_lat[qs[2] + 1, qs[0]])) / 2
        )

        WS_up_leftx = (
            All_ang_lon[ws[0, 1], ws[1, 0]]
            + (All_ang_lon[ws[0, 1], ws[1, 0]] - (All_ang_lon[ws[0, 1], ws[1, 0] + 1]))
            / 2
        )
        WS_up_lefty = (
            All_ang_lat[ws[0, 1], ws[1, 0]]
            - (All_ang_lat[ws[0, 1], ws[1, 0]] - (All_ang_lat[ws[0, 1] - 1, ws[1, 0]]))
            / 2
        )
        WS_up_rightx = (
            All_ang_lon[ws[0, 1], ws[1, 1]]
            - (All_ang_lon[ws[0, 1], ws[1, 1]] - (All_ang_lon[ws[0, 1], ws[1, 1] - 1]))
            / 2
        )
        WS_up_righty = (
            All_ang_lat[ws[0, 1], ws[1, 1]]
            - (All_ang_lat[ws[0, 1], ws[1, 1]] - (All_ang_lat[ws[0, 1] - 1, ws[1, 1]]))
            / 2
        )
        WS_down_rightx = (
            All_ang_lon[ws[0, 0], ws[1, 1]]
            - (All_ang_lon[ws[0, 0], ws[1, 1]] - (All_ang_lon[ws[0, 0], ws[1, 1] - 1]))
            / 2
        )
        WS_down_righty = (
            All_ang_lat[ws[0, 0], ws[1, 1]]
            + (All_ang_lat[ws[0, 0], ws[1, 1]] - (All_ang_lat[ws[0, 0] + 1, ws[1, 1]]))
            / 2
        )
        WS_down_leftx = (
            All_ang_lon[ws[0, 0], ws[1, 0]]
            + (All_ang_lon[ws[0, 0], ws[1, 0]] - (All_ang_lon[ws[0, 0], ws[1, 0] + 1]))
            / 2
        )
        WS_down_lefty = (
            All_ang_lat[ws[0, 0], ws[1, 0]]
            + (All_ang_lat[ws[0, 0], ws[1, 0]] - (All_ang_lat[ws[0, 0] + 1, ws[1, 0]]))
            / 2
        )

    _c = -1
    _nl = len(quentity) // 3 + 2
    mean_pos = []
    fig, axis = plt.subplots(_nl, 3, figsize=(12, ratio * _nl), constrained_layout=True)

    for i in range(len(q)):
        if q[i] == "I":
            _c += 1
            if type(locklist) != type(None) and _c == lockline:
                # print("locking")
                data = p[i, t].copy()
                data[locklist[0] == 1] = 0
            else:
                data = p[i, t].copy()

            minmax = {"interval": AsymmetricPercentileInterval(1, 99)}
            if type(max_I) != type(None) and type(min_I) != type(None):
                minmax = {"vmin": min_I, "vmax": max_I}
            norm = ImageNormalize(data, **minmax, stretch=SqrtStretch())
            im = axis[_c, 0].pcolormesh(
                All_ang_lon, All_ang_lat, data, norm=norm, cmap="magma"
            )
            axis[_c, 0].set_title(
                "Intensity ($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)",
                fontsize=title_size,
            )
            axis[_c, 0].set_xlabel(
                "Helioprojective longitude \n (arcsec)", fontsize=label_size
            )
            axis[_c, 0].set_ylabel(
                "Helioprojective latitude \n (arcsec)", fontsize=label_size
            )
            axis[_c, 0].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 0], extend=("both" if visualize_saturation else None)
            )

            ylabel = axis[_c, 0].yaxis.get_label()
            axis_position = axis[_c, 0].get_position().bounds
            ypos = ylabel.get_position()
            ylabel.set_position(ypos)
            ypos_data = axis[_c, 0].transAxes.inverted().transform(ypos)
            ypos_fig = fig.transFigure.transform(ypos_data)

            print(
                "axis_position",
                axis_position,
                "\n",
                "ypos",
                ypos,
                "\n",
                "ypos_data",
                ypos_data,
                "\n",
                "ypos_fig",
                ypos_fig,
                "\n",
                "texw=t_position",
                axis_position[0] + ypos[0] * (axis_position[2]),
                "\n",
                "texw=t_position",
                axis_position[1] + ypos[1] * (axis_position[3]),
                "\n",
            )
            # print(fig.transFigure.transform(ypos))
            axis[_c, 0].text(
                -0.7,
                0.5,
                ("" if type(linenames) == type(None) else (linenames[_c])),
                bbox={
                    "facecolor": (0.2, 1, 0.4),
                    "alpha": 0.5,
                    "edgecolor": "black",
                    "pad": 2,
                },
                fontsize=sup_title_size,
                rotation=90,
                va="center",
                transform=axis[_c, 0].transAxes,
            )

        if q[i] == "x":

            data, _ = gen_velocity(
                p[i, t], quite_sun=qs, correction=maxwell_correction, verbose=verbose
            )
            if type(locklist) != type(None) and _c == lockline:
                # print("locking")
                data[locklist[0] == 1] = np.nan

            cmap = plt.get_cmap(cmap_doppler).copy()
            (
                cmap.set_extremes(under="yellow", over="green")
                if visualize_saturation
                else 0
            )

            im = axis[_c, 1].pcolormesh(
                All_ang_lon, All_ang_lat, data, vmin=min_V, vmax=max_V, cmap=cmap
            )
            axis[_c, 1].set_title("Doppler (km/s)", fontsize=title_size)
            axis[_c, 1].set_xlabel(
                "Helioprojective longitude \n (arcsec)", fontsize=label_size
            )
            axis[_c, 1].set_ylabel(
                "Helioprojective latitude \n (arcsec)", fontsize=label_size
            )
            axis[_c, 1].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 1], extend=("both" if visualize_saturation else None)
            )
            mean_pos.append(np.nanmean(p[i, t]))
        if q[i] == "s":
            data = p[i, t]
            if type(locklist) != type(None) and _c == lockline:
                # print("locking")
                data[locklist[0] == 1] = np.nan
            cmap = plt.get_cmap("hot").copy()
            (
                cmap.set_extremes(under="green", over="violet")
                if visualize_saturation
                else 0
            )
            im = axis[_c, 2].pcolormesh(
                All_ang_lon, All_ang_lat, data, cmap=cmap, vmax=max_W, vmin=min_W
            )

            axis[_c, 2].set_title("$\sigma$ ($\AA$)", fontsize=title_size)
            axis[_c, 2].set_xlabel(
                "Helioprojective longitude \n (arcsec)", fontsize=label_size
            )
            axis[_c, 2].set_ylabel(
                "Helioprojective latitude \n (arcsec)", fontsize=label_size
            )
            axis[_c, 2].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 2], extend=("both" if visualize_saturation else None)
            )
        if q[i] == "B":
            _c += 1
            minmax = (
                {"interval": AsymmetricPercentileInterval(1, 99)}
                if type(max_B) == type(None)
                else {"vmin": min_B, "vmax": max_B}
            )
            norm = ImageNormalize(p[i, t], **minmax, stretch=SqrtStretch())

            im = axis[_c, 0].pcolormesh(
                All_ang_lon, All_ang_lat, p[i, t], norm=norm, cmap="magma"
            )
            axis[_c, 0].set_title(
                "Background Intensity\n($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)",
                fontsize=title_size,
            )
            axis[_c, 0].set_xlabel(
                "Helioprojective longitude \n (arcsec)", fontsize=label_size
            )
            axis[_c, 0].set_ylabel(
                "Helioprojective latitude \n (arcsec)", fontsize=label_size
            )
            axis[_c, 0].set_aspect("equal")
            plt.colorbar(
                im, ax=axis[_c, 0], extend=("both" if visualize_saturation else None)
            )

            if type(locklist) == type(None):
                axis[_c, 2].remove()
            else:
                vals = [0, 1]
                N = len(vals)
                col_dict = {0.1: (1, 1, 1), 0.9: (0, 0, 0)}
                cmap = ListedColormap([col_dict[x] for x in col_dict.keys()])

                # In case you want the labels to show other than the numbers put the list of these labels here
                labels = ["Free", "Locked"]
                len_lab = len(labels)

                # prepare for the normalizer
                norm_bins = np.sort([*col_dict.keys()]) + 0.5
                # print("sort",norm_bins)
                norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
                # print("insert", norm_bins)
                # Make normalizer and formatter
                norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
                fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

                im = axis[_c, 2].pcolormesh(
                    All_ang_lon, All_ang_lat, locklist[t], cmap=cmap, norm=norm, alpha=1
                )
                axis[_c, 2].set_title("Locking flag matrix", fontsize=title_size)
                axis[_c, 2].set_xlabel(
                    "Helioprojective longitude \n (arcsec)", fontsize=label_size
                )
                axis[_c, 2].set_ylabel(
                    "Helioprojective latitude \n (arcsec)", fontsize=label_size
                )
                diff = norm_bins[1:] - norm_bins[:-1]
                tickz = norm_bins[:-1] + diff / 2
                # cbar = plt.colorbar(im,ax=axis[_c,2],  ticks=tickz,format=fmt)

                # cbar.ax.set_xticklabels(labels)

            if type(convlist) == type(None):
                axis[_c, 1].remove()
            else:
                data_conv = convlist[t, :, :].copy()
                # designing colormap -> color choice and their number
                vals = (np.unique(data_conv)).astype(np.int8)
                N = len(vals)
                n = int(N ** (1 / 3)) + (0 if int(N ** (1 / 3)) == N ** (1 / 3) else 1)
                yn = N ** (1 / 3)
                # print("N,n",N,n)
                col_dict = {}
                # print("vals",vals,'n',n)
                i = 0
                for b in range(n):
                    if i == N - 1:
                        break
                    for g in range(n):
                        if i == N - 1:
                            break
                        for r in range(n):
                            if i == N - 1:
                                break
                            col_dict[vals[i]] = (
                                1 - (r / max(n - 1, 1)),
                                1 - (g / max(n - 1, 1)),
                                1 - (b / max(n - 1, 1)),
                            )
                            i += 1
                # print("col_dict",col_dict)
                # designing colormap -> constructing the cmap
                cmap = ListedColormap([col_dict[x] for x in col_dict.keys()])

                # In case you want the labels to show other than the numbers put the list of these labels here
                labels = vals.copy()
                len_lab = len(labels)

                # prepare for the normalizer
                norm_bins = np.sort([*col_dict.keys()]) + 0.5
                # print("sort",norm_bins)
                norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
                # print("insert", norm_bins)
                # Make normalizer and formatter
                norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
                fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

                im = axis[_c, 1].pcolormesh(
                    All_ang_lon, All_ang_lat, data_conv, cmap=cmap, norm=norm
                )
                axis[_c, 1].set_title("convolution level map", fontsize=title_size)
                axis[_c, 1].set_xlabel(
                    "Helioprojective longitude \n (arcsec)", fontsize=label_size
                )
                axis[_c, 1].set_ylabel(
                    "Helioprojective latitude \n (arcsec)", fontsize=label_size
                )
                axis[_c, 1].set_aspect("equal")

                diff = norm_bins[1:] - norm_bins[:-1]
                tickz = norm_bins[:-1] + diff / 2
                plt.colorbar(im, ax=axis[_c, 1], ticks=tickz)
    if True:
        mn = np.nanmean(
            w[:, :, :, :], axis=(0, 1)  # ws[0,0]:ws[0,1],  # ws[1,0]:ws[1,1]
        )
        minmax = {"interval": AsymmetricPercentileInterval(1, 99)}
        if type(max_I) != type(None) and type(min_I) != type(None):
            minmax = {"vmin": min_I, "vmax": max_I}

        norm = ImageNormalize(mn, **minmax, stretch=SqrtStretch())
        im = axis[-1, 0].pcolormesh(
            All_ang_lon, All_ang_lat, mn, norm=norm, cmap="magma"
        )
        # axis[-1,0].set_aspect('auto')
        plt.colorbar(
            im, ax=axis[-1, 0], extend=("both" if visualize_saturation else None)
        )

        axis[-1, 0].plot(
            [QS_up_leftx, QS_up_rightx, QS_down_rightx, QS_down_leftx, QS_up_leftx],
            [QS_up_lefty, QS_up_righty, QS_down_righty, QS_down_lefty, QS_up_lefty],
            "o-",
            alpha=0.7,
            color=(0, 1, 0),
            lw=ratio / 2,
            label="selected\nquite region",
        )
        axis[-1, 0].plot(
            [WS_up_leftx, WS_up_rightx, WS_down_rightx, WS_down_leftx, WS_up_leftx],
            [WS_up_lefty, WS_up_righty, WS_down_righty, WS_down_lefty, WS_up_lefty],
            "o-",
            alpha=1,
            color=(1, 0, 0),
            lw=ratio / 2,
            label="Analysed region",
        )

        axis[-1, 0].set_title("original averaged\n over spectrum", fontsize=title_size)
        axis[-1, 0].set_xlabel(
            "Helioprojective longitude \n (arcsec)", fontsize=label_size
        )
        axis[-1, 0].set_ylabel(
            "Helioprojective latitude \n (arcsec)", fontsize=label_size
        )
        axis[-1, 1].step(
            sa,
            np.nanmean(
                w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 2, 3)
            ),
        )

        # print(mean_pos)
        # print(sa)
        for x in mean_pos:
            axis[-1, 1].axvline(
                x, ls=":", label="line: {:02d}".format(mean_pos.index(x))
            )
        axis[-1, 1].legend()
        axis[-1, 1].set_title(
            "original average spectrum/\naverage line positions/\n segments",
            fontsize=ratio * 4,
        )

        axis[-1, 2].remove()
        fig.suptitle(
            (
                (suptitle + "\n")
                if type(suptitle) != type(None)
                else "" + kw + " window" + "\n" + raster[kw].meta["DATE_SUN"][:-4]
            ),
            bbox={
                "facecolor": (0.2, 1, 0.4),
                "alpha": 0.5,
                "edgecolor": "black",
                "pad": 2,
            },
            fontsize=sup_title_size,
        )
        if type(segmentation) != type(None):
            if len(segmentation.shape) != 1:
                for seg in segmentation:
                    color = np.random.rand(3)
                    color = 0.8 * color / np.sqrt(np.sum(color**2))
                    axis[-1, 1].axvspan(seg[0], seg[1], alpha=0.5, color=color)
            else:
                seg = segmentation
                color = np.random.rand(3)
                color = 0.8 * color / np.sqrt(np.sum(color**2))
                axis[-1, 1].axvspan(seg[0], seg[1], alpha=0.5, color=color)
        for ax in axis.flatten():
            ax.ticklabel_format(useOffset=False)
            ax.ticklabel_format(useOffset=False)
        # fig.tight_layout()
        if save:
            plt.savefig(filename)

    return fig, axis, All_ang_lon, All_ang_lat


def plot_main(
    raster,
    kwOrOrder,
    paramlist: np.ndarray,
    quentity: list,
    convlist=None,
    locklist=None,
    lockline=None,
    suptitle=None,
    window_size: np.ndarray = np.array([[0, -1], [0, -1]]),
    quite_sun=np.array([0, -1, 0, -1]),
    maxwell_correction=False,
    t: int = 0,
    segmentation=None,
    save=False,
    filename="./imgs/res.jpg",
    min_I=None,
    max_I=None,
    min_V=-50,
    max_V=+50,
    min_W=None,
    max_W=None,
    min_B=None,
    max_B=None,
    visualize_saturation=True,
    verbose=0,
):
    unq = spu.unique_windows(raster)
    if type(kwOrOrder) == str:
        kw = kwOrOrder
    else:
        kw = unq[kwOrOrder]

    r = raster
    w = r[kw].data
    q = quentity
    p = paramlist
    ws = window_size
    sa = r[kw].spectral_axis * 10**10
    qs = quite_sun
    cmap_doppler = "twilight_shifted"
    ratio = 4
    cbar_tick_size = 6 * ratio
    title_size = 8 * ratio
    label_size = 7 * ratio
    legend_size = 5 * ratio
    sup_title_size = 15 * ratio
    if True:  # Coordinate matrix
        All_ang_lat = r[kw].celestial.data.lat.arcsec
        All_ang_lon = r[kw].celestial.data.lon.arcsec

        All_ang_lon[All_ang_lon >= 180 * 3600] = (
            All_ang_lon[All_ang_lon >= 180 * 3600] - 360 * 3600
        )
        All_ang_lat[All_ang_lat >= 180 * 3600] = (
            All_ang_lat[All_ang_lat >= 180 * 3600] - 360 * 3600
        )

    if True:  # Adjusting quite sun
        if qs[1] < 0:
            if ws[1, 1] < 0:
                qs[1] = w.shape[3] + 2 + ws[1, 1] + qs[1]
            else:
                qs[1] = ws[1, 1] + 1 + qs[1]
        if qs[3] == -1:
            if ws[0, 1] < 0:
                qs[3] = w.shape[2] + 2 + ws[0, 1] + qs[3]
            else:
                qs[3] = ws[0, 1] + 1 + qs[3]

    if True:  # quite sun and window size coordinates
        QS_up_leftx = (
            All_ang_lon[qs[3], qs[0]]
            + (All_ang_lon[qs[3], qs[0]] - (All_ang_lon[qs[3], qs[0] + 1])) / 2
        )
        QS_up_lefty = (
            All_ang_lat[qs[3], qs[0]]
            - (All_ang_lat[qs[3], qs[0]] - (All_ang_lat[qs[3] - 1, qs[0]])) / 2
        )
        QS_up_rightx = (
            All_ang_lon[qs[3], qs[1]]
            - (All_ang_lon[qs[3], qs[1]] - (All_ang_lon[qs[3], qs[1] - 1])) / 2
        )
        QS_up_righty = (
            All_ang_lat[qs[3], qs[1]]
            - (All_ang_lat[qs[3], qs[1]] - (All_ang_lat[qs[3] - 1, qs[1]])) / 2
        )
        QS_down_rightx = (
            All_ang_lon[qs[2], qs[1]]
            - (All_ang_lon[qs[2], qs[1]] - (All_ang_lon[qs[2], qs[1] - 1])) / 2
        )
        QS_down_righty = (
            All_ang_lat[qs[2], qs[1]]
            + (All_ang_lat[qs[2], qs[1]] - (All_ang_lat[qs[2] + 1, qs[1]])) / 2
        )
        QS_down_leftx = (
            All_ang_lon[qs[2], qs[0]]
            + (All_ang_lon[qs[2], qs[0]] - (All_ang_lon[qs[2], qs[0] + 1])) / 2
        )
        QS_down_lefty = (
            All_ang_lat[qs[2], qs[0]]
            + (All_ang_lat[qs[2], qs[0]] - (All_ang_lat[qs[2] + 1, qs[0]])) / 2
        )

        WS_up_leftx = (
            All_ang_lon[ws[0, 1], ws[1, 0]]
            + (All_ang_lon[ws[0, 1], ws[1, 0]] - (All_ang_lon[ws[0, 1], ws[1, 0] + 1]))
            / 2
        )
        WS_up_lefty = (
            All_ang_lat[ws[0, 1], ws[1, 0]]
            - (All_ang_lat[ws[0, 1], ws[1, 0]] - (All_ang_lat[ws[0, 1] - 1, ws[1, 0]]))
            / 2
        )
        WS_up_rightx = (
            All_ang_lon[ws[0, 1], ws[1, 1]]
            - (All_ang_lon[ws[0, 1], ws[1, 1]] - (All_ang_lon[ws[0, 1], ws[1, 1] - 1]))
            / 2
        )
        WS_up_righty = (
            All_ang_lat[ws[0, 1], ws[1, 1]]
            - (All_ang_lat[ws[0, 1], ws[1, 1]] - (All_ang_lat[ws[0, 1] - 1, ws[1, 1]]))
            / 2
        )
        WS_down_rightx = (
            All_ang_lon[ws[0, 0], ws[1, 1]]
            - (All_ang_lon[ws[0, 0], ws[1, 1]] - (All_ang_lon[ws[0, 0], ws[1, 1] - 1]))
            / 2
        )
        WS_down_righty = (
            All_ang_lat[ws[0, 0], ws[1, 1]]
            + (All_ang_lat[ws[0, 0], ws[1, 1]] - (All_ang_lat[ws[0, 0] + 1, ws[1, 1]]))
            / 2
        )
        WS_down_leftx = (
            All_ang_lon[ws[0, 0], ws[1, 0]]
            + (All_ang_lon[ws[0, 0], ws[1, 0]] - (All_ang_lon[ws[0, 0], ws[1, 0] + 1]))
            / 2
        )
        WS_down_lefty = (
            All_ang_lat[ws[0, 0], ws[1, 0]]
            + (All_ang_lat[ws[0, 0], ws[1, 0]] - (All_ang_lat[ws[0, 0] + 1, ws[1, 0]]))
            / 2
        )

    N = len(quentity) // 3
    for num_ions in range(N):
        _nl = 3
        mean_pos = []
        # fig, axis = plt.subplots(_nl,3, figsize=(ratio*3,ratio*3),constrained_layout=True)
        # conv_c = 0
        # plt.rcParams.update({'font.size': 30})
        fig = plt.figure(figsize=(24, 24))
        gs0 = fig.add_gridspec(2, 2)
        gs1 = gs0[0, 0].subgridspec(2, 1)
        gs2 = gs1[1].subgridspec(1, 2)

        # fig, axis = plt.subplots(2,2, figsize=(24,8.2*_nl))

        for i in range(3):
            if q[i] == "I":
                axI = fig.add_subplot(gs0[0, 1])

                if type(locklist) != type(None) and num_ions == lockline:
                    # print("locking")
                    data = p[i + 3 * num_ions, t].copy()
                    data[locklist[0] == 1] = 0
                else:
                    data = p[i + 3 * num_ions, t].copy()
                minmax = {"interval": AsymmetricPercentileInterval(1, 99)}

                if type(max_I) == list:
                    Max_I = max_I[num_ions]
                else:
                    Max_I = max_I
                if type(min_I) == list:
                    Min_I = min_I[num_ions]
                else:
                    Min_I = min_I

                if type(Max_I) != type(None) or type(Min_I) != type(None):
                    print("Min_I", Min_I, "Max_I", Max_I)
                    minmax = {"vmin": Min_I, "vmax": Max_I}

                norm = ImageNormalize(data, **minmax, stretch=SqrtStretch())
                im = axI.pcolormesh(
                    All_ang_lon, All_ang_lat, data, norm=norm, cmap="magma"
                )
                axI.set_title(
                    "Intensity ($W \cdot m^{-2} \cdot sr^{-1}\cdot nm^{-1}$)",
                    fontsize=title_size,
                )
                axI.set_xlabel(
                    "Helioprojective longitude \n (arcsec)", fontsize=label_size
                )
                axI.set_ylabel(
                    "Helioprojective latitude \n (arcsec)", fontsize=label_size
                )
                axI.set_aspect("equal")
                axI.xaxis.set_tick_params(labelsize=label_size)
                axI.yaxis.set_tick_params(labelsize=label_size)

                cbar = plt.colorbar(
                    im,
                    shrink=0.75,
                    ax=axI,
                    extend=("both" if visualize_saturation else None),
                )
                cbar.ax.tick_params(labelsize=cbar_tick_size)

            if q[i] == "x":
                axx = fig.add_subplot(gs0[1, 0])
                data, _ = gen_velocity(
                    p[i + 3 * num_ions, t],
                    quite_sun=qs,
                    correction=maxwell_correction,
                    verbose=verbose,
                )
                if type(locklist) != type(None) and num_ions == lockline:
                    # print("locking")
                    data[locklist[0] == 1] = np.nan
                mean_pos.append(np.nanmean(p[i + 3 * num_ions, t]))
                cmap = plt.get_cmap(cmap_doppler).copy()
                (
                    cmap.set_extremes(under="yellow", over="green")
                    if visualize_saturation
                    else 0
                )

                # print("Calculated mean: ", mean_x)
                im = axx.pcolormesh(
                    All_ang_lon, All_ang_lat, data, vmin=min_V, vmax=max_V, cmap=cmap
                )

                axx.set_title("Doppler (km/s)", fontsize=title_size)
                axx.set_xlabel(
                    "Helioprojective longitude \n (arcsec)", fontsize=label_size
                )
                axx.set_ylabel(
                    "Helioprojective latitude \n (arcsec)", fontsize=label_size
                )
                axx.set_aspect("equal")
                axx.xaxis.set_tick_params(labelsize=label_size)
                axx.yaxis.set_tick_params(labelsize=label_size)

                cbar = plt.colorbar(
                    im,
                    shrink=0.75,
                    ax=axx,
                    extend=("both" if visualize_saturation else None),
                )
                cbar.ax.tick_params(labelsize=cbar_tick_size)
            if q[i] == "s":
                data = p[i + 3 * num_ions, t]
                if type(locklist) != type(None) and num_ions == lockline:
                    # print("locking")
                    data[locklist[0] == 1] = np.nan

                axs = fig.add_subplot(gs0[1, 1])
                cmap = plt.get_cmap("hot").copy()
                (
                    cmap.set_extremes(under="green", over="violet")
                    if visualize_saturation
                    else 0
                )
                im = axs.pcolormesh(
                    All_ang_lon,
                    All_ang_lat,
                    p[i + 3 * num_ions, t],
                    cmap=cmap,
                    vmax=max_W,
                    vmin=min_W,
                )

                axs.set_title("$\sigma$ ($\AA$)", fontsize=title_size)
                axs.set_xlabel(
                    "Helioprojective longitude \n (arcsec)", fontsize=label_size
                )
                axs.set_ylabel(
                    "Helioprojective latitude \n (arcsec)", fontsize=label_size
                )
                axs.set_aspect("equal")
                axs.xaxis.set_tick_params(labelsize=label_size)
                axs.yaxis.set_tick_params(labelsize=label_size)

                cbar = plt.colorbar(
                    im,
                    shrink=0.75,
                    ax=axs,
                    extend=("both" if visualize_saturation else None),
                )
                cbar.ax.tick_params(labelsize=cbar_tick_size)

        mn = np.nanmean(
            w[:, :, :, :], axis=(0, 1)  # ws[0,0]:ws[0,1],  # ws[1,0]:ws[1,1]
        )
        axAI = fig.add_subplot(gs2[0])

        minmax = {
            "interval": AsymmetricPercentileInterval(1, 99),
            "vmin": None,
            "vmax": None,
        }
        # if type(Max_I)!=type(None) and type(Min_I)!=type(None):s

        norm = ImageNormalize(mn, **minmax, stretch=SqrtStretch())
        im = axAI.pcolormesh(All_ang_lon, All_ang_lat, mn, norm=norm, cmap="magma")
        # axAI.set_aspect('auto')
        cbar = plt.colorbar(
            im, shrink=0.75, ax=axAI, extend=("both" if visualize_saturation else None)
        )
        cbar.ax.tick_params(labelsize=cbar_tick_size)

        axAI.plot(
            [QS_up_leftx, QS_up_rightx, QS_down_rightx, QS_down_leftx, QS_up_leftx],
            [QS_up_lefty, QS_up_righty, QS_down_righty, QS_down_lefty, QS_up_lefty],
            "o-",
            alpha=1,
            color=(0, 1, 0),
            lw=ratio / 2,
            label="Selected\nquite region",
        )
        axAI.plot(
            [WS_up_leftx, WS_up_rightx, WS_down_rightx, WS_down_leftx, WS_up_leftx],
            [WS_up_lefty, WS_up_righty, WS_down_righty, WS_down_lefty, WS_up_lefty],
            "o-",
            alpha=1,
            color=(1, 0, 0),
            lw=ratio / 2,
            label="Analysed region",
        )

        axAI.set_title("Average intensity", fontsize=title_size)
        axAI.set_xlabel("Helioprojective longitude \n (arcsec)", fontsize=label_size)
        axAI.set_ylabel("Helioprojective latitude \n (arcsec)", fontsize=label_size)
        axAI.set_aspect("equal")
        axAI.xaxis.set_tick_params(labelsize=label_size)
        axAI.yaxis.set_tick_params(labelsize=label_size)

        axAS = fig.add_subplot(gs2[1])
        axAS.step(
            sa,
            np.nanmean(
                w[:, :, ws[0, 0] : ws[0, 1], ws[1, 0] : ws[1, 1]], axis=(0, 2, 3)
            ),
        )

        # print(mean_pos)
        # print(sa)
        for x in mean_pos:
            axAS.axvline(x, ls=":", label="line: {:02d}".format(mean_pos.index(x)))
        axAS.legend(fontsize=legend_size)
        axAS.set_title("Average spectrum", fontsize=title_size)
        axAS.set_ylabel("Intensity", fontsize=label_size)
        axAS.set_xlabel("Wavelength $(\AA)$", fontsize=label_size)
        axAS.xaxis.set_tick_params(labelsize=label_size)
        axAS.yaxis.set_tick_params(labelsize=label_size)

        fig.text(
            0.04,
            0.80,
            (suptitle[num_ions] if type(suptitle) != type(None) else kw)
            + "\n"
            + raster[kw].meta["DATE_SUN"][:-4],
            fontsize=sup_title_size,
            bbox={
                "facecolor": (0.2, 1, 0.4),
                "alpha": 0.5,
                "edgecolor": "black",
                "pad": 2,
            },
        )
        if type(segmentation) != type(None):
            if len(segmentation.shape) != 1:
                for seg in segmentation:
                    color = np.random.rand(3)
                    color = 0.8 * color / np.sqrt(np.sum(color**2))
                    axAS.axvspan(seg[0], seg[1], alpha=0.5, color=color)
            else:
                seg = segmentation
                color = np.random.rand(3)
                color = 0.8 * color / np.sqrt(np.sum(color**2))
                axAS.axvspan(seg[0], seg[1], alpha=0.5, color=color)
        plt.tight_layout()
        if save:
            plt.savefig(filename.format(num_ions))

    # return fig,[axI,axx,axs,axAI,axAS],All_ang_lon, All_ang_lat,ang_lon, ang_lat,quite_sun
