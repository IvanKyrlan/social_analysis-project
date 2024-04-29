import csv
import os

def parse_file(filename, position, take):
    csv.field_size_limit(5000000)
    script_dir = os.path.dirname(os.path.abspath(__file__)) + '\\data'
    file_path = os.path.join(script_dir, filename)
    text_list = []

    with open(file_path, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        i = 0
        for row in csv_reader:
            if i < take:
                text_list.append(row[position])
                i += 1
            else: 
                break

    return text_list


def parse_file_param(filename, position, take, position_param, param):
    csv.field_size_limit(5000000)
    script_dir = os.path.dirname(os.path.abspath(__file__)) + '\\data'
    file_path = os.path.join(script_dir, filename)
    text_list = []

    with open(file_path, 'r', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file)
        i = 0
        for row in csv_reader:
            if (i < take) and (row[position_param] == param):
                text_list.append(row[position])
                i += 1
            elif (i >= take): 
                break
            else:
                continue

    return text_list