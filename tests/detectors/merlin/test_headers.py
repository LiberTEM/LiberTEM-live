import numpy as np

from libertem_live.detectors.merlin.data import AcquisitionHeader, FrameHeader


ACQUISITION_HEADER_RAW = r"""HDR,
Time and Date Stamp (day, mnth, yr, hr, min, s):	18/05/2020 16:51:48
Chip ID:	W529_F5,-,-,-
Chip Type (Medipix 3.0, Medipix 3.1, Medipix 3RX):	Medipix 3RX
Assembly Size (NX1, 2X2):	   1x1
Chip Mode  (SPM, CSM, CM, CSCM):	SPM
Counter Depth (number):	6
Gain:	SLGM
Active Counters:	Alternating
Thresholds (keV):	1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0
DACs:	175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511
bpc File:	c:\MERLIN_Quad_Config\W529_F5\W529_F5_SPM.bpc,,,
DAC File:	c:\MERLIN_Quad_Config\W529_F5\W529_F5_SPM.dacs,,,
Gap Fill Mode:	Distribute
Flat Field File:	None
Dead Time File:	Dummy (C:\<NUL>\)
Acquisition Type (Normal, Th_scan, Config):	Normal
Frames in Acquisition (Number):	16384
Frames per Trigger (Number):	128
Trigger Start (Positive, Negative, Internal):	Rising Edge LVDS
Trigger Stop (Positive, Negative, Internal):	Internal
Sensor Bias (V):	120 V
Sensor Polarity (Positive, Negative):	Positive
Temperature (C):	Board Temp 37.384918 Deg C
Humidity (%):	Board Humidity 1.331848
Medipix Clock (MHz):	120MHz
Readout System:	Merlin Quad
Software Version:	0.67.0.9
End	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               """  # noqa: W291,E501


def test_parse_acquisition_header():
    header = AcquisitionHeader.from_raw(ACQUISITION_HEADER_RAW.encode('latin1'))

    assert header.frames_in_acquisition == 128*128
    assert header.frames_per_trigger == 128


FRAME_HEADER_RAW = r"""MQ1,000001,00384,01,0256,0256,U08,   1x1,01,2020-05-18 16:51:49.971626,0.000555,0,0,0,1.200000E+2,5.110000E+2,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,0.000000E+0,3RX,175,511,000,000,000,000,000,000,125,255,125,125,100,100,082,100,087,030,128,004,255,129,128,176,168,511,511,MQ1A,2020-05-18T14:51:49.971626178Z,555000ns,6
"""  # noqa


def test_parse_frame_header():
    header = FrameHeader.from_raw(FRAME_HEADER_RAW.encode("latin1"))
    assert header.header_size_bytes == 384
    assert header.dtype == np.uint8
    assert header.mib_dtype == "u08"
    assert header.bits_per_pixel == 6
    assert header.image_size == (256, 256)
    assert header.image_size_eff == (256, 256)
    assert header.image_size_bytes == 256 * 256
    assert header.sequence_first_image == 1
    assert header.num_chips == 1
