import os, json, sys
from glob import glob


def main(data_dir):
    """Write data.json listing every NIfTI in data_dir under the 'val' key."""
    data_dir = data_dir.strip()
    nii_filelist = sorted(glob(os.path.join(data_dir, '*.nii*')))

    out_json = os.path.join(data_dir, 'data.json')

    data_json = {'val': []}

    for nii_file in nii_filelist:
        nii_base = os.path.basename(nii_file)
        data_json['val'].append({'image': nii_base})

    json_object = json.dumps(data_json, indent=4)

    print(f"JSON input: {out_json}")
    with open(out_json, "w") as outfile:
        outfile.write(json_object)
    return out_json


if __name__ == '__main__':
    main(sys.argv[1])
