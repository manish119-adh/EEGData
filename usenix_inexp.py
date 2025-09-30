
import csv

from globals import EEGEvent

def event_list_generator(filename):
    '''
    Generate events from the file
    '''
    
    events=[]
    with open(filename, mode='r') as file:
        csv_reader = csv.reader(file, delimiter="\t")
        header = False
        
        for row in csv_reader:
            if header: # skip title row
                events.append(EEGEvent(timestamp=float(row[0]), event_label=row[1].split("-")[0]))
            header = True
        
    return events


if __name__ == "__main__":
    eventfile = "data/USENIX_Inexpensive/studie001_2019.05.08_10.15.34-events.txt"
    event_list = event_list_generator(eventfile)
    print(event_list)



            
    