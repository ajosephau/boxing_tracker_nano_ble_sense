import json
import os

files = {"left.json": ["lbodyshot.json", "lcross.json", "lhook.json", "lupper.json"],
         "right.json": ["rbodyshot.json", "rjab.json", "rhook.json", "rupper.json"]}

for key, value_list in files.items():
    print(key)
    index = 0
    strokes_list = []
    for value in value_list:
        print(value)
        with open(value, 'r') as infile:
            strokes = json.load(infile)
            label = value.split(".")[0][1:]
            for stroke in strokes['strokes']:
                stroke_to_add = {"index": index, "label": label, "strokePoints":stroke["strokePoints"]}
                strokes_list.append(stroke_to_add)
                index = index + 1
    output_data = {"strokes": strokes_list}
    with open(key, 'w') as outfile:
        json.dump(output_data, outfile)
