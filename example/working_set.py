"""
Example of usage prepare_working_set function.

Please change input_file and output_file as you want.
"""
from credo_cf import prepare_working_set

input_file = '/tmp/credo/detections/export_1584805204914_1585394157807.json'
output_file = '/tmp/out.json'

prepare_working_set(input_file, output_file)
