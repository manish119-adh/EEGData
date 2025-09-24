
import csv

from globals import EEGEvent

def event_list_generator(filename):
    '''
    Generate events from the file
    '''
    def gen():
        events=[]
        with open(filename, mode='r', newline='') as file:
            csv_reader = csv.reader(file, delimiter="\t")
            header = False
            
            for row in csv_reader:
                if header: # skip title row
                    events.append(EEGEvent(timestamp=row[0], event_label=row[1].split("-")[0]))
                header = True
        yield events
    return gen
            
    