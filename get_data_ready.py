import os, csv, sys, glob, shutil


if os.path.isdir('example_run/formatted_data'):
    shutil.rmtree('example_run/formatted_data')
os.mkdir('example_run/formatted_data')



peps = []
tcrs = []
with open('example_run/data/train.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[1],line[2],line[3],line[4],line[5],line[6]]))
with open('example_run/data/val.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[1],line[2],line[3],line[4],line[5],line[6]]))
with open('example_run/data/test.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[1],line[2],line[3],line[4],line[5],line[6]]))

peps = list(set(peps))
tcrs = list(set(tcrs))
pep_seq_to_id = {}
tcr_seq_to_id = {}

with open('example_run/formatted_data/ids_pep.csv', 'w') as fw:
    for en, pep in enumerate(peps):
        fw.write('pep%d,%s\n'%(en,pep))
        pep_seq_to_id[pep] = 'pep%d'%en
        
with open('example_run/formatted_data/ids_tcr.csv', 'w') as fw:
    for en, tcr in enumerate(tcrs):
        fw.write('tcr%d,%s\n'%(en,tcr))
        tcr_seq_to_id[tcr] = 'tcr%d'%en
    
    
    
        

peps, tcrs, labels, split = [],[],[],[]
    
with open('example_run/data/train.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[1],line[2],line[3],line[4],line[5],line[6]])])
        labels.append(int(line[-1]))
        split.append('train')

with open('example_run/data/val.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[1],line[2],line[3],line[4],line[5],line[6]])])
        labels.append(int(line[-1]))
        split.append('val')

with open('example_run/data/test.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[1],line[2],line[3],line[4],line[5],line[6]])])
        labels.append(int(line[-1]))
        split.append('test')
    
with open('example_run/formatted_data/data.csv', 'w') as fw:
    fw.write('pep_id,tcr_id,label,split\n')
    for aa,bb,cc,dd in zip(peps, tcrs, labels, split):
        fw.write('%s,%s,%d,%s\n'%(aa,bb,cc,dd))



        