import os
import pdb

save = []
filename = './train.txt'

tokens = ['_member_of_domain_usage']

new_filename = './train_new.txt'
entity_set = set()

with open(filename, 'r') as f:
	lines = f.read().split('\n')
	for line in lines:
		for token in tokens:
			if token in line:
				save.append(line)
				entity = line.split('\t')
				head, tail = entity[0], entity[2]
				entity_set.add(head)
				entity_set.add(tail)
				break

with open(new_filename, 'w') as f:
	for line in save:
		f.write(line + '\n')

count = 0
with open('new_entities.dict', 'w') as f:
	for e in entity_set:
		f.write('{}\t{}\n'.format(count, e))
		count += 1

count = 0
with open('new_relations.dict', 'w') as f:
	for r in tokens:
		f.write('{}\t{}\n'.format(count, r))
		count += 1




