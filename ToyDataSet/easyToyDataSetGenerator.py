
from music21 import *
import os
environment.set('musicxmlPath', '/usr/bin/mscore')
environment.set('midiPath', '/usr/bin/mscore')


rootname = "clean_midi"
filenames = os.listdir(rootname)


for filename in filenames:

    midifile = rootname + "/" + filename + "/" + \
        os.listdir(rootname+"/"+filename)[0]
    midiname = os.listdir(rootname+"/"+filename)[0]
    littleMelody = converter.parse(midifile)

    chords = littleMelody.chordify()

    chords = chords.elements

    listToWrite = []

    time = 0
    time_write = 0
    for mychord in chords:
        if type(mychord) != chord.Chord:
            # do nothing
            print("do nothing because it's type :", type(mychord))
        elif mychord.isChord:
            time += mychord.quarterLength
            while time_write < time:
                root = mychord.root().name
                if root == "E-":
                    root = "D#"
                if root == "B-":
                    root = "A#"
                name = mychord.commonName
                if name[0:5] == "minor":
                    harmo = "min"
                elif name[0:5] == "major":
                    harmo = "maj"
                else:
                    harmo = "N"
                if harmo == "N":
                    listToWrite.append("N")
                else:
                    listToWrite.append(str(time_write)+" "+root+":"+harmo)
                time_write += 1

    with open('generated/'+midiname, 'w') as f:
        for item in listToWrite:
            f.write("%s\n" % item)
