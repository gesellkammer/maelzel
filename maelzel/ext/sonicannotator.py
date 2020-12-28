import bpf4
import shutil
import tempfile
import subprocess
from pathlib import Path
import os
import csv

from typing import Optional as Opt


def _find_sonic_annotator() -> Opt[str]:
    return shutil.which("sonic-annotator")

# Obtained via sonic-annotator -s vamp:pyin:pyin:smoothedpitchtrack > pyin.n3

_rdf_pyin_smooth_pitch = """
@prefix xsd:      <http://www.w3.org/2001/XMLSchema#> .
@prefix vamp:     <http://purl.org/ontology/vamp/> .
@prefix :         <#> .

:transform a vamp:Transform ;
    vamp:plugin <http://vamp-plugins.org/rdf/plugins/pyin#pyin> ;
    vamp:step_size "{stepsize}"^^xsd:int ; 
    vamp:block_size "{fftsize}"^^xsd:int ; 
    vamp:plugin_version \"\"\"2\"\"\" ; 
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "lowampsuppression" ] ;
        vamp:value "{lowampsuppression}"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "onsetsensitivity" ] ;
        vamp:value "{onsetsensitivity}"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "outputunvoiced" ] ;
        vamp:value "0"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "precisetime" ] ;
        vamp:value "0"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "prunethresh" ] ;
        vamp:value "{prunethresh}"^^xsd:float ;
    ] ;
    vamp:parameter_binding [
        vamp:parameter [ vamp:identifier "threshdistr" ] ;
        vamp:value "{threshdistr}"^^xsd:float ;
    ] ;
    vamp:output <http://vamp-plugins.org/rdf/plugins/pyin#pyin_output_smoothedpitchtrack> .
"""


def pyin_smooth_pitch(sndfile:str, fftsize=2048, stepsize=256,
                      lowampsuppression=0.1, threshdistr=2.,
                      onsetsensitivity=0.7,
                      prunethresh=0.1) -> bpf4.core.Linear:
    sonic = _find_sonic_annotator()
    if sonic is None:
        raise RuntimeError("sonic-annotator was not found, install it from "
                           "https://code.soundsoftware.ac.uk/projects/sonic-annotator/files")

    rdfstr = _rdf_pyin_smooth_pitch.format(fftsize=fftsize, stepsize=stepsize,
                                           lowampsuppression=lowampsuppression,
                                           threshdistr=threshdistr,
                                           onsetsensitivity=onsetsensitivity,
                                           prunethresh=prunethresh)
    rdf = tempfile.mktemp(prefix="pyin-smoothpitch", suffix=".n3")
    open(rdf, "w").write(rdfstr)
    subprocess.call([sonic, "-t", rdf, sndfile, "-w", "csv", "--csv-force"])
    base = os.path.splitext(os.path.split(sndfile)[1])[0]
    generated_basefile = f"{base}_vamp_pyin_pyin_smoothedpitchtrack.csv"
    outfile = Path(sndfile).parent / generated_basefile
    if not outfile.exists():
        raise RuntimeError("Expected output file {outfile} not found")

    ts = []
    freqs = []

    for row in csv.reader(open(str(outfile))):
        ts.append(float(row[0]))
        freqs.append(float(row[1]))

    return bpf4.core.Linear(ts, freqs)