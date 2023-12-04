import os, csv, sys, glob, shutil


if os.path.isdir('formatted_data'):
    shutil.rmtree('formatted_data')
os.mkdir('formatted_data')



peps = []
tcrs = []
with open('example_data/train.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[2],line[3],line[4],line[5],line[6],line[7]]))
with open('example_data/val.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[2],line[3],line[4],line[5],line[6],line[7]]))
with open('example_data/test.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(line[0])
        tcrs.append('_'.join([line[2],line[3],line[4],line[5],line[6],line[7]]))

peps = list(set(peps))
tcrs = list(set(tcrs))
pep_seq_to_id = {}
tcr_seq_to_id = {}

with open('formatted_data/ids_pep.csv', 'w') as fw:
    for en, pep in enumerate(peps):
        fw.write('pep%d,%s\n'%(en,pep))
        pep_seq_to_id[pep] = 'pep%d'%en
        
with open('formatted_data/ids_tcr.csv', 'w') as fw:
    for en, tcr in enumerate(tcrs):
        fw.write('tcr%d,%s\n'%(en,tcr))
        tcr_seq_to_id[tcr] = 'tcr%d'%en
    
    
    
        

peps, tcrs, labels, split = [],[],[],[]
    
with open('example_data/train.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[2],line[3],line[4],line[5],line[6],line[7]])])
        labels.append(int(line[-1]))
        split.append('train')

with open('example_data/val.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[2],line[3],line[4],line[5],line[6],line[7]])])
        labels.append(int(line[-1]))
        split.append('val')

with open('example_data/test.csv', 'r') as f:
    r = csv.reader(f)
    next(r)
    for line in r:
        peps.append(pep_seq_to_id[line[0]])
        tcrs.append(tcr_seq_to_id['_'.join([line[2],line[3],line[4],line[5],line[6],line[7]])])
        labels.append(int(line[-1]))
        split.append('test')
    
with open('formatted_data/data.csv', 'w') as fw:
    fw.write('pep_seq,tcr_seq,label,split\n')
    for aa,bb,cc,dd in zip(peps, tcrs, labels, split):
        fw.write('%s,%s,%d,%s\n'%(aa,bb,cc,dd))


