import argparse
import glog
import os
import json

if __name__ == "__main__":
    """ Parse invalid JSON output to the valid one. """
    CLI = argparse.ArgumentParser(
        description='Provide the input folder and output folder')

    CLI.add_argument(
        '-i',
        '--input',
        default="final_pns",
        help='Path to the folder with SecondNet output. We expect files to have *.pn extension')

    CLI.add_argument(
        '-o',
        '--output',
        default="final_pns_json",
        help='Path to the output folder with valid JSON files')

    ARGS = CLI.parse_args()

    input_path = ARGS.input
    output_path = ARGS.output
    all_files = os.listdir(input_path)
    input_files = []
    glog.debug(all_files)

    for afile in all_files:
        if afile.endswith('.pn'):
            input_files.append(afile)
    glog.debug(input_files)
    
    for pn_file in input_files:
        input_file = os.path.join(input_path, pn_file)
        with open(input_file, 'r') as f:
            broken_json = f.read().replace(' ','')
            broken_json = broken_json.replace('\"final\":', '')
            broken_json = broken_json.replace('],]', ']]')
            broken_json = broken_json.replace('],}', ']}')
            broken_json = broken_json.replace('\'', '\"')
            glog.debug('broken_json = {}'.format(broken_json))
            valid_json = json.loads(broken_json)
    
        glog.debug('valid_json = {}'.format(valid_json))
        out_fname = os.path.join(output_path, '{}.json'.format(pn_file))
        with open(out_fname, 'w') as outfile:
            json.dump(valid_json, outfile, indent=4)

        glog.info('converted {} to {}'.format(input_file, out_fname))

    glog.info('done, see folder: {}'.format(output_path))
