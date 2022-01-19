import os

event = 'EVENT_1131'
data_path = '/home/laghi/src/cosmolisa/data/EMRIs/EMRI_SAMPLE_MODEL101_TTOT10yr_SIG2_GAUSS'
path = '/home/laghi/src/cosmolisa/data/EMRIs/FAKE_CATALOGS'

all_files  = os.listdir(data_path)
selected_event = [f for f in all_files if event in f]

os.system('mkdir -p FAKE_CATALOGS/M1_{0}_1_COPIES'.format(event))
os.system('cp -r {data}/{sel} FAKE_CATALOGS/M1_{event}_1_COPIES/{sel}_1'.format(data=data_path, sel=selected_event[0], event=event))

os.system('mkdir -p FAKE_CATALOGS/M1_{0}_10_COPIES'.format(event))
for k in range(1,11):
    os.system('cp -r {data}/{sel} FAKE_CATALOGS/M1_{event}_10_COPIES/{sel}_{num}'.format(data=data_path, sel=selected_event[0], event=event, num=k))

os.system('mkdir -p FAKE_CATALOGS/M1_{0}_20_COPIES'.format(event))
for k in range(1,21):
    os.system('cp -r {data}/{sel} FAKE_CATALOGS/M1_{event}_20_COPIES/{sel}_{num}'.format(data=data_path, sel=selected_event[0], event=event, num=k))

os.system('mkdir -p FAKE_CATALOGS/M1_{0}_30_COPIES'.format(event))
for k in range(1,31):
    os.system('cp -r {data}/{sel} FAKE_CATALOGS/M1_{event}_30_COPIES/{sel}_{num}'.format(data=data_path, sel=selected_event[0], event=event, num=k))

os.system('mkdir -p FAKE_CATALOGS/M1_{0}_40_COPIES'.format(event))
for k in range(1,41):
    os.system('cp -r {data}/{sel} FAKE_CATALOGS/M1_{event}_40_COPIES/{sel}_{num}'.format(data=data_path, sel=selected_event[0], event=event, num=k))