#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""

import dask
import datetime
import os
import namelist
import numpy as np
import xarray as xr

from dask.distributed import LocalCluster, Client
from util import input, mat
from thermo import thermo

def get_fn_thermo():
    fn_th = '%s/thermo_%s_%d%02d_%d%02d.nc' % (namelist.output_directory, namelist.exp_prefix,
                                               namelist.start_year, namelist.start_month,
                                               namelist.end_year, namelist.end_month)
    return(fn_th)


def _mid_level_setup(ds_ta, ds_hus):
    """Return (lvl, lvl_d, p_midlevel, lvl_mid) with pressure reindexing applied."""
    lvl   = ds_ta[input.get_lvl_key()]
    lvl_d = np.copy(lvl.data)

    ta  = ds_ta[input.get_temp_key()]
    hus = ds_hus[input.get_sp_hum_key()]

    # Ensure lowest model level (highest pressure) is first.
    if (lvl[0] - lvl[1]) < 0:
        ta  = ta.reindex( {input.get_lvl_key(): lvl[::-1]})
        hus = hus.reindex({input.get_lvl_key(): lvl[::-1]})
        lvl_d = lvl_d[::-1]

    p_midlevel = namelist.p_midlevel                  # Pa
    if lvl.units in ['millibars', 'hPa']:
        lvl_d     *= 100                              # convert to Pa
        p_midlevel = namelist.p_midlevel / 100        # hPa for .sel()

    lvl_mid = lvl.sel({input.get_lvl_key(): p_midlevel}, method='nearest')
    return ta, hus, lvl_d, p_midlevel, lvl_mid


def compute_rh_mid_and_denom(dt_start, dt_end):
    """Lightweight first pass: compute rh_mid and chi denominator (spss - sps).

    Skips the expensive CAPE_PI computation.  Called before the main
    compute_thermo run whenever a climatological experiment is requested, so
    that full-period means of rh_mid and chi_denom can be derived and passed
    back into compute_thermo.
    """
    ds_sst = input.load_sst(dt_start, dt_end).load()
    ds_psl = input.load_mslp(dt_start, dt_end).load()
    ds_ta  = input.load_temp(dt_start, dt_end).load()
    ds_hus = input.load_sp_hum(dt_start, dt_end).load()
    lon_ky = input.get_lon_key()
    lat_ky = input.get_lat_key()
    sst_ky = input.get_sst_key()

    ta, hus, lvl_d, p_midlevel, lvl_mid = _mid_level_setup(ds_ta, ds_hus)

    nTime     = len(ds_sst['time'])
    rh_mid    = np.zeros(ds_psl[input.get_mslp_key()].shape)
    chi_denom = np.zeros(ds_psl[input.get_mslp_key()].shape)

    p_midlevel_Pa = float(lvl_mid) * 100 if lvl_mid.units in ['millibars', 'hPa'] else float(lvl_mid)

    for i in range(nTime):
        sst_interp = mat.interp_2d_grid(ds_sst[lon_ky], ds_sst[lat_ky],
                                        np.nan_to_num(ds_sst[sst_ky][i, :, :].data),
                                        ds_ta[lon_ky], ds_ta[lat_ky])
        if 'C' in ds_sst[sst_ky].units:
            sst_interp = sst_interp + 273.15

        psl          = ds_psl[input.get_mslp_key()][i, :, :]
        ta_midlevel  = ta[i].sel( {input.get_lvl_key(): p_midlevel}, method='nearest').data
        hus_midlevel = hus[i].sel({input.get_lvl_key(): p_midlevel}, method='nearest').data

        rh_mid[i, :, :]    = thermo.conv_q_to_rh(ta_midlevel, hus_midlevel, p_midlevel_Pa)
        chi_denom[i, :, :] = thermo.chi_denominator(sst_interp, psl.data,
                                                     ta_midlevel, p_midlevel_Pa, hus_midlevel)
    return (rh_mid, chi_denom)


def compute_thermo(dt_start, dt_end, rh_mid_climo=None, chi_denom_climo=None):
    """Compute vmax, chi, and rh_mid for the period [dt_start, dt_end].

    Parameters
    ----------
    rh_mid_climo : np.ndarray (nlat, nlon) or None
        When provided, chi is computed using climatological mid-level RH:
        rv_climo = rh_mid_climo * rs(T, pm) replaces the actual specific
        humidity in the entropy numerator.  The rh_mid output is also set
        to rh_mid_climo (same 2-D field broadcast over every time step) so
        that chi and rh_mid remain consistent.
    chi_denom_climo : np.ndarray (nlat, nlon) or None
        When provided, the chi denominator (spss - sps) is replaced by this
        pre-computed climatological value, consistent with a climatological
        potential intensity.  If rh_mid_climo is also given the same rv_climo
        is used for the numerator, giving a fully self-consistent climo chi.
    """
    ds_sst = input.load_sst(dt_start, dt_end).load()
    ds_psl = input.load_mslp(dt_start, dt_end).load()
    ds_ta  = input.load_temp(dt_start, dt_end).load()
    ds_hus = input.load_sp_hum(dt_start, dt_end).load()
    lon_ky = input.get_lon_key()
    lat_ky = input.get_lat_key()
    sst_ky = input.get_sst_key()

    ta, hus, lvl_d, p_midlevel, lvl_mid = _mid_level_setup(ds_ta, ds_hus)

    nTime  = len(ds_sst['time'])
    vmax   = np.zeros(ds_psl[input.get_mslp_key()].shape)
    chi    = np.zeros(ds_psl[input.get_mslp_key()].shape)
    rh_mid = np.zeros(ds_psl[input.get_mslp_key()].shape)

    p_midlevel_Pa = float(lvl_mid) * 100 if lvl_mid.units in ['millibars', 'hPa'] else float(lvl_mid)

    use_climo_chi = (rh_mid_climo is not None) or (chi_denom_climo is not None)

    for i in range(nTime):
        sst_interp = mat.interp_2d_grid(ds_sst[lon_ky], ds_sst[lat_ky],
                                        np.nan_to_num(ds_sst[sst_ky][i, :, :].data),
                                        ds_ta[lon_ky], ds_ta[lat_ky])
        if 'C' in ds_sst[sst_ky].units:
            sst_interp = sst_interp + 273.15

        psl          = ds_psl[input.get_mslp_key()][i, :, :]
        ta_i         = ta[i]
        hus_i        = hus[i]

        # TODO: Check units of psl, ta, and hus
        vmax_args = (sst_interp, psl.data, lvl_d, ta_i.data, hus_i.data)
        vmax[i, :, :] = thermo.CAPE_PI_vectorized(*vmax_args)

        ta_midlevel  = ta_i.sel( {input.get_lvl_key(): p_midlevel}, method='nearest').data
        hus_midlevel = hus_i.sel({input.get_lvl_key(): p_midlevel}, method='nearest').data

        if use_climo_chi:
            # sat_deficit_climo handles all four combinations of rh_mid_climo /
            # chi_denom_climo being None or not.
            chi_raw = thermo.sat_deficit_climo(sst_interp, psl.data,
                                               ta_midlevel, p_midlevel_Pa,
                                               hus_midlevel,
                                               rh_climo=rh_mid_climo,
                                               chi_denom_climo=chi_denom_climo)
            chi[i, :, :] = np.minimum(np.maximum(chi_raw, 0), 10)
            # rh_mid output: climatological value when rh_mid_climo is given,
            # actual value otherwise (e.g. climo-vmax-only experiment).
            if rh_mid_climo is not None:
                rh_mid[i, :, :] = rh_mid_climo
            else:
                rh_mid[i, :, :] = thermo.conv_q_to_rh(ta_midlevel, hus_midlevel,
                                                        p_midlevel_Pa)
        else:
            chi_args = (sst_interp, psl.data, ta_midlevel, p_midlevel_Pa, hus_midlevel)
            chi[i, :, :]    = np.minimum(np.maximum(thermo.sat_deficit(*chi_args), 0), 10)
            rh_mid[i, :, :] = thermo.conv_q_to_rh(ta_midlevel, hus_midlevel, p_midlevel_Pa)

    return (vmax, chi, rh_mid)


def gen_thermo():
    # TODO: Assert all of the datasets have the same length in time.
    if os.path.exists(get_fn_thermo()):
        return

    use_climo_rh   = getattr(namelist, 'climo_rh_mid', False)
    use_climo_vmax = getattr(namelist, 'climo_vmax',   False)

    # Load dataset metadata.
    dt_start, dt_end = input.get_bounding_times()
    ds = input.load_mslp()

    ct_bounds = [dt_start, dt_end]
    ds_times = input.convert_from_datetime(ds,
                   np.array([x for x in input.convert_to_datetime(ds, ds['time'].values)
                             if x >= ct_bounds[0] and x <= ct_bounds[1]]))

    n_chunks = namelist.n_procs
    chunks   = np.array_split(ds_times, np.minimum(n_chunks, np.floor(len(ds_times) / 2)))
    cl_args  = {'n_workers': namelist.n_procs, 'processes': True, 'threads_per_worker': 1}

    # ------------------------------------------------------------------
    # Pass 1 (lightweight, only when a climo experiment is requested):
    # compute rh_mid and chi_denom for every time step – no CAPE_PI.
    # Take their full-period means to obtain the climatological arrays.
    # ------------------------------------------------------------------
    rh_mid_climo    = None
    chi_denom_climo = None

    if use_climo_rh or use_climo_vmax:
        print('Pass 1: computing rh_mid and chi_denom for climatology...')
        lazy_p1 = []
        with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
            for i in range(n_chunks):
                lazy_p1.append(
                    dask.delayed(compute_rh_mid_and_denom)(chunks[i][0], chunks[i][-1]))
            out_p1 = dask.compute(*lazy_p1, scheduler='processes', num_workers=n_chunks)

        rh_mid_all    = np.concatenate([x[0] for x in out_p1], axis=0)  # (nT, nlat, nlon)
        chi_denom_all = np.concatenate([x[1] for x in out_p1], axis=0)

        if use_climo_rh:
            rh_mid_climo = rh_mid_all.mean(axis=0)      # (nlat, nlon)
            print('rh_mid climatology: global mean = %.4f' % float(rh_mid_climo.mean()))
        if use_climo_vmax:
            chi_denom_climo = chi_denom_all.mean(axis=0)
            print('chi_denom climatology: global mean = %.4f' % float(chi_denom_climo.mean()))

    # ------------------------------------------------------------------
    # Pass 2 (full computation): vmax via CAPE_PI + chi + rh_mid.
    # Climo arrays (None when not requested) are forwarded to each worker.
    # ------------------------------------------------------------------
    lazy_p2 = []
    with LocalCluster(**cl_args) as cluster, Client(cluster) as client:
        for i in range(n_chunks):
            lazy_p2.append(
                dask.delayed(compute_thermo)(chunks[i][0], chunks[i][-1],
                                             rh_mid_climo, chi_denom_climo))
        out = dask.compute(*lazy_p2, scheduler='processes', num_workers=n_chunks)

    # Clean up: ensure monthly timestamps land on the 15th of each month.
    ds_times = input.convert_from_datetime(ds,
                  np.array([datetime.datetime(x.year, x.month, 15) for x in
                           [x for x in input.convert_to_datetime(ds, ds['time'].values)
                            if x >= ct_bounds[0] and x <= ct_bounds[1]]]))

    vmax   = np.concatenate([x[0] for x in out], axis=0)
    chi    = np.concatenate([x[1] for x in out], axis=0)
    rh_mid = np.concatenate([x[2] for x in out], axis=0)

    # ------------------------------------------------------------------
    # Climo-vmax post-processing: replace transient vmax with its time
    # mean (broadcast over all time steps).  chi already used the climo
    # denominator, so chi and vmax are now both climatological.
    # ------------------------------------------------------------------
    if use_climo_vmax:
        vmax_climo = vmax.mean(axis=0, keepdims=True)   # (1, nlat, nlon)
        vmax = np.broadcast_to(vmax_climo, vmax.shape).copy()
        print('vmax climatology: global mean = %.2f m/s' % float(vmax_climo.mean()))

    ds_thermo = xr.Dataset(data_vars = dict(vmax   = (['time', 'lat', 'lon'], vmax),
                                            chi    = (['time', 'lat', 'lon'], chi),
                                            rh_mid = (['time', 'lat', 'lon'], rh_mid)),
                           coords    = dict(lon  = ('lon',  ds[input.get_lon_key()].data),
                                            lat  = ('lat',  ds[input.get_lat_key()].data),
                                            time = ('time', ds_times.astype('datetime64[ns]'))))
    ds_thermo.to_netcdf(get_fn_thermo())
    print('Saved %s' % get_fn_thermo())
