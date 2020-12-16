from __future__ import absolute_import
TRANSP_STOP      = mmc(0x01)
TRANSP_PLAY      = mmc(0x02)
TRANSP_FF        = mmc(0x04)
TRANSP_REW       = mmc(0x05)
TRANSP_REC       = mmc(0x06)
TRANS_REC_STOP   = mmc(0x07)
TRANSP_PAUSE     = mmc(0x09)
TRANSP_MMC_RESET = mmc(0x0D)

def mmc_goto(hours=0, minutes=0, secs=0, frames=0):
    """
    dont use this function, use MidiMessage.midiMachineControlGoto directly

    this is here only for documentation.
    """
    from rtmidi2 import MidiMessage
    return MidiMessage.midiMachineControlGoto(hours, minutes, secs, frames)