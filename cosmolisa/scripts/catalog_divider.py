import os, random

# Script to generate splitted catalogs.
# M1 has 180 events, M6 has 260 events.

M1_tot = 180
M6_tot = 260

# Set the catalog flag and the number of parts (2 or 3) in which you want to split the catalog
#FIXME: generalize to arbitrary number of parts
catalog_flag = 'M1'
number_of_parts = 2

if catalog_flag == 'M1': 
    tot_number = M1_tot
elif (catalog_flag == 'M6'):
    tot_number = M6_tot
else:
    print("Please set a correct catalog flag.")

size = int(tot_number / number_of_parts)  

if (number_of_parts == 2):
    parts = {"SUBSAMPLE1": [], "SUBSAMPLE2": []}

    samples    = [int(x) for x in range(1001,1001+tot_number)]
    parts["SUBSAMPLE1"] = random.sample(samples, size)
    # subsample1 = subsample1.sort()
    parts["SUBSAMPLE2"] = [x for x in samples if x not in parts["SUBSAMPLE1"]]
    # subsample2 = subsample2.sort()

elif (number_of_parts == 3):
    parts = {"SUBSAMPLE1": [], "SUBSAMPLE2": [], "SUBSAMPLE3": []}

    samples    = [int(x) for x in range(1001,1001+tot_number)]
    parts["SUBSAMPLE1"] = random.sample(samples, size)
    # subsample1 = subsample1.sort()
    samples2   = [x for x in samples if x not in parts["SUBSAMPLE1"]]
    parts["SUBSAMPLE2"] = random.sample(samples2, size)
    # subsample2 = subsample2.sort()
    parts["SUBSAMPLE3"] = [x for x in samples2 if x not in parts["SUBSAMPLE2"]]
    # subsample3 = subsample2.sort()

total_length = 0
for key,part in parts.items():
    print("#event in {}: {}".format(key, len(part)))
    total_length += len(part)
    part = part.sort()
print("Total:", total_length)



if catalog_flag == 'M1': 
    path = '/home/laghi/src/cosmolisa/data/EMRIs/EMRI_SAMPLE_MODEL101_TTOT10yr_SIG2_GAUSS'
elif (catalog_flag == 'M6'):
    path = '/home/laghi/src/cosmolisa/data/EMRIs/EMRI_SAMPLE_MODEL106_TTOT10yr_SIG2_GAUSS'

all_files  = os.listdir(path)
all_events = [f for f in all_files if "EVENT" in f]

for key,value in parts.items():
    print('\nCreating {0}/{0}_SPLITTED_INTO_{1}_{2}'.format(catalog_flag, number_of_parts, key))
    os.system('mkdir -p {0}/{0}_SPLITTED_INTO_{1}_{2}'.format(catalog_flag, number_of_parts, key))
    print(key, value)
    for v in value:
        for f in all_events:
            if f.count(str(v)):
                os.system('cp -r {4}/{3} {0}/{0}_SPLITTED_INTO_{1}_{2}'.format(catalog_flag, number_of_parts, key, f, path)) 

