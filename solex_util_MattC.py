import numpy as np
import re
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_sun, AltAz
import astropy.units as u
import math

def parse_value(raw_value):
    """Parse a single raw string into a normalized representation
       while preserving the original for round-tripping."""
    raw_value = raw_value.strip()

    # --- Version numbers: treat as strings ---
    if re.match(r"^\d+(\.\d+)+$", raw_value):
        return {"parsed": {"string": raw_value}, "raw": raw_value}

    # Compound values (split into multiple key/value pairs logically)
    # Example: "1936x64" → {width: 1936, height: 64}
    if re.match(r"^\d+x\d+$", raw_value):
        w, h = map(int, raw_value.split("x"))
        return {"parsed": {"width": w, "height": h}, "raw": raw_value}

    # Numbers with units
    match = re.match(r"^([0-9.+-]+)\s*(ms|s|fps|°|C)?$", raw_value, re.I)
    if match:
        val, unit = match.groups()
        num = float(val) if "." in val or "e" in val.lower() else int(val)
        return {"parsed": {"value": num, "unit": unit}, "raw": raw_value}

    # ISO8601 datetime
    try:
        dt = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        return {"parsed": {"datetime": dt}, "raw": raw_value}
    except Exception:
        pass

    # Plain number
    if re.match(r"^-?\d+(\.\d+)?$", raw_value):
        num = float(raw_value) if "." in raw_value else int(raw_value)
        return {"parsed": {"value": num}, "raw": raw_value}

    # Fallback: store as plain string
    return {"parsed": {"string": raw_value}, "raw": raw_value}


def parse_settings_file(path):
    """Read settings file into a dict with both parsed and raw values."""
    settings = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Section headers [something]
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            settings["_section"] = section
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        settings[key.strip()] = parse_value(value.strip())

    return settings


def reverse_convert(settings):
    """Rebuild the original file from the dictionary (lossless)."""
    lines = []
    if "_section" in settings:
        lines.append(f"[{settings['_section']}]")
    for key, val in settings.items():
        if key == "_section":
            continue
        # Always use the raw version for round-trip
        lines.append(f"{key}={val['raw']}")
    return "\n".join(lines)

def sun_angular_diameter(midcapture_datetime, latitude_deg, longitude_deg, altitude_m=0):
    """
    Calculate the topocentric apparent angular diameter of the Sun in arcseconds.

    Parameters:
    -----------
    midcapture_datetime : datetime.datetime
        Timestamp of observation (from settings file, e.g., MidCapture).
    latitude_deg : float
        Observer latitude in degrees (positive north).
    longitude_deg : float
        Observer longitude in degrees (positive east, negative west).
    altitude_m : float, optional
        Observer altitude in meters. Default is 0.

    Returns:
    --------
    float
        Topocentric apparent angular diameter of the Sun in arcseconds.
    """
    # 1️⃣ Observer location
    observer = EarthLocation(lat=latitude_deg*u.deg,
                             lon=longitude_deg*u.deg,
                             height=altitude_m*u.m)

    # 2️⃣ Convert timestamp to Astropy Time
    t = Time(midcapture_datetime)

    # 3️⃣ Sun geocentric coordinates
    sun = get_sun(t)

    # 4️⃣ Transform to AltAz (topocentric) frame
    sun_topocentric = sun.transform_to(AltAz(obstime=t, location=observer))
    distance_km = sun_topocentric.distance.to(u.km).value

    # 5️⃣ Compute apparent angular diameter
    R_sun_km = 696_340  # Sun radius
    theta_rad = 2 * math.atan(R_sun_km / distance_km)
    theta_arcsec = math.degrees(theta_rad) * 3600

    return theta_arcsec

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import os

    path = "/Users/mconsidine@middlebury.edu/Downloads/2025-08-17-1340_9-Sunvt.CameraSettings.txt"

    if os.path.exists(path):
        settings = parse_settings_file(path)

        # test to access dictionary
        ##width = settings["Capture Area"]["parsed"]["width"]
        ##print(f"Width : {width}")

        # determine target info
        apparent_solar_diam = sun_angular_diameter(settings["MidCapture"]["parsed"]["datetime"],
                                             43,-73, 200)
        print(f"Topocentric Sun diameter: {apparent_solar_diam:.2f} arcseconds")

        # Add a new entry to dictionary
        settings["tgt_diam_effective"] = {
            "parsed": {"value": f"{apparent_solar_diam:.2f}"},  # normalized/parsed value for Python use
            "raw": f"{apparent_solar_diam:.2f}"  # raw string to use when writing back
        }

        Camera_dictionary = {"ASI1600":{"sensor_pixels":4656,"pixel_size":3.8,"camera_binning":2},
                             "ASI174":{"sensor_pixels":1926,"pixel_size":5.86,"camera_binning":1}}

        SHG_dictionary = {"Hale":{"clear_ap":3*25.4,
                                  "native_focal_length":18*12*25.4,
                                  "barlow_power":1,
                                  "shg_collim_fl":13*12*25.4,
                                  "shg_camera_fl":13*12*25.4,
                                  "shg_slit_length":19,
                                  "shg_slit_width":0.003*25.4*1000,
                                  "shg_grating_lpmm":1200,
                                  },
                          "MLAstro_MattC":{"clear_ap":102,
                                           "native_focal_length":663,
                                           "barlow_power":1,
                                           "shg_collim_fl":72,
                                           "shg_camera_fl":72,
                                           "shg_slit_length":7,
                                           "shg_slit_width":7,
                                           "shg_grating_lpmm":2400,
                                           }}

        #print(f"Test 1 : {SHG_dictionary["Hale"]["scan_mult"]}")
        #print(f"Test 2 : {SHG_dictionary["MLAstro_MattC"]["wvlength"]}")

        SHG = "Hale" #MLAstro_MattC
        CAMERA = "ASI1600" #ASI174

        # mount info
        scan_mult = 1.0 #multiple of sidereal rate
        # wavelength
        wvlength = 656.28 #nm
        # target
        tgt_diam_effective = apparent_solar_diam
        # scope info
        clear_ap = SHG_dictionary[SHG]["clear_ap"] #mm
        native_focal_length = SHG_dictionary[SHG]["native_focal_length"] #663 #mm
        barlow_power = SHG_dictionary[SHG]["barlow_power"]
        focal_length = native_focal_length * barlow_power
        scope_f_ratio = focal_length / clear_ap
        # camera info
        sensor_pixels = Camera_dictionary[CAMERA]["sensor_pixels"]
        pixel_size = Camera_dictionary[CAMERA]["pixel_size"]
        sensor_length = sensor_pixels * pixel_size/1000 #mm
        camera_binning = Camera_dictionary[CAMERA]["camera_binning"]
        ideal_sampling = 4 #pixels per airy disc
        sensor_pixels = sensor_pixels / camera_binning
        pixel_size = pixel_size * camera_binning
        # shg info
        shg_collim_fl = SHG_dictionary[SHG]["shg_collim_fl"]
        shg_camera_fl = SHG_dictionary[SHG]["shg_camera_fl"]
        shg_magn = shg_camera_fl/shg_collim_fl
        shg_slit_length = SHG_dictionary[SHG]["shg_slit_length"]
        shg_slit_width = SHG_dictionary[SHG]["shg_slit_width"]
        shg_grating_lpmm = SHG_dictionary[SHG]["shg_grating_lpmm"]
        # sw reconstruction info; EXPERIMENTAL
        sw_sampling = 1

        scan_speed = scan_mult*(360*60*60)/(24*60*60) #arcseconds per second
        print(f"\nscan speed : {scan_speed} arcseconds per second")

        tgt_scan_time = tgt_diam_effective/scan_speed #seconds
        print(f"target scan time : {tgt_scan_time} seconds")

        scope_resolution_rayleigh = 1.22*(wvlength*10e-9)*(clear_ap*10e-3)*(180/np.pi)*3600
        print(f"scope resolution : {scope_resolution_rayleigh} arcsecs")

        ##ideal_fps = scan_speed/scope_resolution_rayleigh
        ##ideal_fps = ideal_fps * ideal_sampling
        ###ideal_frames = tgt_diam_effective/scope_resolution_rayleigh
        ##ideal_frames = tgt_scan_time/ideal_fps

        ##print(f"\nideal fps : {ideal_fps}")
        ##print(f"ideal frames : {ideal_frames}")

        scope_camera_spatial_resolution = pixel_size * (180*3600/(np.pi*1000))/focal_length
        print(f"\nsensor_resolution : {scope_camera_spatial_resolution} arcsecs per pixel")
        # this should be 3-7 vs 1????
        print(f"nominal sampling ratio (ideal scope/scope-camera); want 3-7 : {scope_resolution_rayleigh/scope_camera_spatial_resolution}")

        print(f"\nideal fps 2: {scan_speed/scope_camera_spatial_resolution}")
        print(f"ideal frames 2: {tgt_diam_effective/scope_camera_spatial_resolution}")

        tgt_diam_effective_mm = (tgt_diam_effective/(180*3600/(np.pi))) * focal_length
        print(f"\ntgt diam : {tgt_diam_effective_mm} mm")
        tgt_slit_coverage = shg_slit_length / tgt_diam_effective_mm
        print(f"slit coverage : {tgt_slit_coverage}*100 (ratio)")
        tgt_required_sweeps = 1 if tgt_slit_coverage > 1.0 else int(1/(0.9*tgt_slit_coverage))+1
        print(f"required sweeps : {tgt_required_sweeps}")

        tgt_scan_frames = tgt_scan_time * settings["ActualFrameRate"]["parsed"]["value"]
        print(f"\n  From video file supplied:")
        print(f"estimated actual target frames : {tgt_scan_frames}")
        tgt_scan_resolution = tgt_diam_effective/tgt_scan_frames #arcseconds
        print(f"target actual scan resolution : {tgt_scan_resolution} arcseconds")
        print(f"actual scan resolution ratio (ideal/actual; >1=need higher fps) : {scope_resolution_rayleigh/tgt_scan_resolution}")
        print(f"actual resolution ratio (want 1?) : {tgt_scan_resolution/scope_camera_spatial_resolution}")

        #for k, v in list(settings.items())[:8]:
        #    print(f"{k}: {v}")
        print(settings)
        # Rebuild the file text
        rebuilt = reverse_convert(settings)

        #print("\nLossless round-trip check:",
        #      open(path, "r", encoding="utf-8").read().strip() == rebuilt.strip())

        #print(rebuilt)
