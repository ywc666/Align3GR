
from __future__ import print_function

import os

import time

import random
import multiprocessing as mp

import numpy as np

#inp is the file name of orignal data file
#train the the file path to write in used to trian
# test the the file path to write in used to test
# number is the user number to testest, number)
def cut(_input,item_file,user_id_file,item_id_file,inter_file,_train,_item_vec):
    user_behav = dict()

    import json
    file = open(_input, 'r') 
    path = _input.split('/')
    path = '/'.join(path[:-1])



    # with open(item_file,'r') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         meta_dict[data['asin']] = data

    with open(user_id_file,'r') as f:
        user_dict = json.load(f)
    user_ids = list(user_dict.keys())
    with open(item_id_file,'r') as f:
        item_dict = json.load(f)
    with open(inter_file,'r') as f:
        user_behav = json.load(f)


    # for [Amazon]
    # for line in file.readlines():
    #     dic = json.loads(line)
    #     datas.append(dic)



    # # for [MIND]
    # with open(_input,'r') as f:
    #     for idx, line in enumerate(f):
    #         print(idx)
    #         # Amazon: user_id | item_id | rating | timestamp
    #         if idx == 0:
    #             continue
    #         arr = line.split('\t')
    #         # if len(arr) != 5:
    #         #     break
    #         if arr[1] not in item_ids:
    #             item_ids.add(arr[1])
    #         if arr[0] not in user_behav:
    #             user_ids.append(arr[0])
    #             user_behav[arr[0]] = list()
    #
    #         line = line.replace("\t", ",")
    #         line = line.strip("\n")
    #         user_behav[arr[0]].append(line)

    random.shuffle(user_ids)
    train_user_ids = user_ids

    #write train data set
    # with open(_train, 'w') as f:
    #     for uid in train_user_ids:
    #         for line in user_behav[user_dict[uid]]:
    #             f.write(line + os.linesep)

    # re label usr id and item id which make the initial id is 0

    
    # 编号从0开始

    print('user number {}, item number {}'.format(len(user_dict),len(item_dict)))
    # train_user_ids={id:i for i,id in enumerate(train_user_ids)}
    # train_item_his = dict()
    # with open(_input,'r') as f:
    #     for line in f:
    #         arr = line.split(',')
    #         if len(arr) != 5:
    #             break   
    #         item = item_ids[int(arr[1])]
    #         if item not in train_item_his:
    #             train_item_his[item] = list()
    #         if int(arr[0]) not in test_user_ids and int(arr[0]) not in validation_user_ids:
    #             #print(1) 
    #             train_item_his[item].append(train_user_ids[int(arr[0])])
    # train_item_vec = np.zeros([len(item_ids), len(train_user_ids)])
    # for i in range(len(item_ids)):
    #     item_his = train_item_his[i]
    #     train_item_vec[i][item_his] = 1.0
    
    # np.save(_item_vec,train_item_vec)
    if os.path.exists(path + '/user_dict.npy') and os.path.exists(path + '/item_dict.npy'):
        user_dict = np.load(path + '/user_dict.npy',allow_pickle=True).tolist()
        item_dict = np.load(path + '/item_dict.npy',allow_pickle=True).tolist()
    else:
        np.save(path + '/user_dict', user_dict) #
        np.save(path + '/item_dict', item_dict) # 
    # exit(0)
    return user_dict, item_dict

def _read(raw_file,test_record_num,item_file,user_id_file,item_id_file,inter_file):
    train_data_file,test_data_file,validation_data_file='train.dat','test.dat','validation.dat'
    path_seg=raw_file.split('/')
    prefix=''
    if len(path_seg)>1  :
        for seg in path_seg[:-1]:
                prefix+=seg+'/'


    user_ids,item_ids=cut(raw_file, item_file,user_id_file,item_id_file,inter_file,\
                        prefix+'raw_'+train_data_file,
                        prefix + 'train_item_vec')

    behavior_dict = dict()  # record the behavior type, 5 types, key is the type, value is the id of the type
    train_sample = dict()  # record user id, item id, cate_id, behavior_id, timestampe. key the liks 'USERID', value is a array.
    test_sample = dict()
    validation_sample=dict()
    user_id = list()
    item_id = list()
    cat_id = list()
    behav_id = list()
    timestamp = list()
    import json

    start = time.time()
    itobj = zip([prefix+'raw_'+train_data_file, prefix+'raw_'+test_data_file, prefix+'raw_'+validation_data_file], [train_sample, test_sample,validation_sample])
    for filename, sample in itobj:
        with open(filename, 'r') as f: 
            for line in f:
                arr = json.loads(line)
                if arr['asin'] not in item_ids.keys(): 
                    continue
                # [amazon]
                user_id.append(user_ids[arr['reviewerID']]) 
                item_id.append(item_ids[arr['asin']]) 
                cat_id.append(0) 
                # if arr[2] not in behavior_dict:
                #     i = len(behavior_dict)
                #     behavior_dict[arr[2]] = i
                behav_id.append(0) 
                timestamp.append(0) 
            sample["USERID"] = np.array(user_id)
            sample["ITEMID"] = np.array(item_id)
            sample["CATID"] = np.array(cat_id)
            sample["BEHAV"] = np.array(behav_id)
            sample["TS"] = np.array(timestamp)

            user_id = []
            item_id = []
            cat_id = []
            behav_id = []
            timestamp = []

    #write train data set
    '''
    with open(prefix+'processed_'+train_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(train_sample["USERID"],train_sample["ITEMID"],
                                                             train_sample["CATID"],train_sample["BEHAV"],train_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')

    with open(prefix+'processed_'+test_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(test_sample["USERID"],test_sample["ITEMID"],
                                                             test_sample["CATID"],test_sample["BEHAV"],test_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')

    with open(prefix+'processed_'+validation_data_file, 'w') as f:
        for user_id,item_id,cat_id,behav_id,timestamp in zip(validation_sample["USERID"],validation_sample["ITEMID"],
                                                             validation_sample["CATID"],validation_sample["BEHAV"],validation_sample["TS"]):

            f.write(str(user_id)+','+str(item_id)+','+str(cat_id)+','+str(behav_id)+','+str(timestamp)+'\n')
    '''
    print("Read data done, {} train records, {} test records"", elapsed: {}".format(len(train_sample["USERID"]),
                                                                                    len(test_sample["USERID"]),
                                                                                    time.time() - start))
    os.remove(prefix + 'raw_' + train_data_file)
    os.remove(prefix + 'raw_' + test_data_file)
    os.remove(prefix + 'raw_' + validation_data_file)
    # train_sample record user id, item id, cate_id, behavior_id, timestampe. key the liks 'USERID', value is a array.
    # test_sample is like train_sample
    # behavior_dict record the behavior type, 5 types, key is the type, value is the id of the type
    return behavior_dict, train_sample, test_sample,validation_sample,len(user_ids),len(item_ids), user_ids

def _gen_user_his_behave(train_sample,mode='user'):

    result = dict()
    iterobj = zip(train_sample["USERID"],train_sample["ITEMID"], train_sample["TS"])
    if mode == 'item': # 
        for user_id, item_id, ts in iterobj:
        # 
            if user_id not in result:
                result[user_id] = list()
            result[user_id].append((item_id, ts))
    
    elif mode == 'user': 

        for user_id, item_id, ts in iterobj:
            if item_id not in result:
                result[item_id] = list()
            result[item_id].append((user_id, ts))

    # aaa = user_his_behav[9863]
    for _, value in result.items(): #
        value.sort(key=lambda x: x[1])
    # user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    
    return result

def _gen_discriminator_samples(train_sample,discriminator_instances_file,train_sample_seg_cnt=400,\
                               parall=5,seq_len=70,min_seq_len=6):
    user_his_behav = _gen_user_his_behave(train_sample)
    path = discriminator_instances_file.split('/')
    path = '/'.join(path[:-1])
    # save user_his_behav to file
    np.save(path + '/user_his_behav', user_his_behav)

    print("user_his_behav len: {}".format(len(user_his_behav)))
    users = list(user_his_behav.keys())
    process = []
    pipes = []
    job_size = int(len(user_his_behav) / parall)
    if len(user_his_behav) % parall != 0:
        parall += 1
    for i in range(parall):
        #a, b = mp.Pipe()
        #pipes.append(a)
        p = mp.Process(
            target=_partial_gen_discriminator_samples,
            args=(users[i * job_size: (i + 1) * job_size],
                  user_his_behav,
                  '{}.part_{}'.format(discriminator_instances_file, i),seq_len,min_seq_len)
        )
        process.append(p)
        p.start()
    '''
    t=0
    for pipe in pipes:
        (count) = pipe.recv()
        t+=count
    print('{} discriminator training instances!'.format(t))
    '''
    for p in process:
        p.join()
    #print('total instances is {}'.format(count))
    # Merge partial files
    with open(discriminator_instances_file, 'w') as f:
        for i in range(parall):
            filename = '{}.part_{}'.format(discriminator_instances_file, i)
            with open(filename, 'r') as f1:
                f.write(f1.read())

            os.remove(filename)

    # Split train sample to segments
    _split_train_sample(discriminator_instances_file,train_sample_seg_cnt)



def _partial_gen_discriminator_samples(users,user_his_behav, filename,seq_len,min_seq_len):
    # user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    # filename is the file to be written into, i.e. sample file
    count = 0
    with open(filename, 'w') as f:
        for user in users:
            value = user_his_behav[user]  # the clicked item id for user
            count = len(value)  # the item number of the user to click
            if count < min_seq_len:
                continue
            arr = [-1 for i in range(seq_len - min_seq_len)] + [v[0] for v in value]
            # arr is [0,0...,0,itemid0,itemid1,....]
            # each line user_id|item1,item2,...,item(self.lem-1)|label
            for i in range(len(arr) - seq_len+1):
                sample = arr[i: i + seq_len-1]
                f.write('{}|'.format(user))  # sample id
                f.write("{}|".format(",".join([str(v) for v in sample])))
                #f.write("{}".format(sample[-1]))  # label, no ts
                f.write("{}".format(",".join([str(v) for v in arr[i+seq_len-1:i+seq_len-1+int(seq_len/5)]])))
                f.write('\n')
                count += 1
    #pipe.send((count))

def _partial_gen_train_sample(users, user_his_behav, filename,seq_len,min_seq_len, pipe):
    #user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    #filename is the file to be written into, i.e. sample file
    stat = dict()# record the frequency of the each item 's appearance
    count=0
    with open(filename, 'w') as f:
        for user in users:
            value = user_his_behav[user]# the clicked item id for user
            length = len(value)# the item number of the user to click
            if length < min_seq_len:
                continue
            arr = [-1 for i in range(seq_len - min_seq_len)] + [v for v in value]   
            #arr is [0,0...,0,itemid0,itemid1,....]
            #each line user_id|item1,item2,...,item(self.lem-1)|label

            for i in range(len(arr) - seq_len + 1 - 2): 
                sample = arr[i: i + seq_len] 
                f.write('{}|'.format(user))  # sample id
                f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                f.write("{}".format(sample[-1]))  # label, no ts
                f.write('\n')
                count+=1
                if sample[-1] not in stat:
                    stat[sample[-1]] = 0
                stat[sample[-1]] += 1
    pipe.send((stat,count))

def _gen_train_sample(train_sample_seg_cnt,train_instances_file,parall=5,seq_len=70,min_seq_len=6,user_his_behav = None):
    #user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp

    path = train_instances_file.split('/')
    path = '/'.join(path[:-1])

    user_his_behav = {int(k): v for k, v in user_his_behav.items()}
    np.save(path + '/his_behav', user_his_behav)
    print("his_behav len: {}".format(len(user_his_behav)))


    users = list(user_his_behav.keys())
    process = []
    pipes = []
    job_size = int(len(user_his_behav) / parall)
    if len(user_his_behav) % parall != 0:
        parall += 1
    for i in range(parall):
        a, b = mp.Pipe()
        pipes.append(a)
        p = mp.Process( target=_partial_gen_train_sample,
            args=(users[i * job_size: (i + 1) * job_size], user_his_behav, '{}.part_{}'.format(train_instances_file, i),seq_len,min_seq_len,b)
        )
        process.append(p)
        p.start()

    stat = dict()# record the frequency of the each item 's appearance
    t=0
    for pipe in pipes:
        (st,count) = pipe.recv()
        t+=count
        for k, v in st.items():
            if k not in stat:
                stat[k] = 0
            stat[k] += v

    for p in process:
        p.join()
    # print('total instances is {}'.format(count))
    # Merge partial files
    with open(train_instances_file, 'w') as f:
        for i in range(parall):
            filename = '{}.part_{}'.format(train_instances_file, i)
            with open(filename, 'r') as f1:
                f.write(f1.read())

            os.remove(filename)

        if False and test_sample is not None:
            user_his_behav = _gen_user_his_behave(train_sample)#
            for user, value in user_his_behav.items():
                length = len(value)  # the item number of the user to click
                if length < min_seq_len:
                    continue
                arr = [-1 for i in range(seq_len - min_seq_len)] + \
                      [v[0] for v in value]
                for i in range(len(arr) - seq_len,len(arr) - seq_len + 1):
                    sample = arr[i: i + seq_len]
                    f.write('{}|'.format(user))  # sample id
                    f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                    f.write("{}".format(sample[-1]))  # label, no ts
                    f.write('\n')


                # if len(value)/2 + 1 < min_seq_len:
                #     continue
                #
                # #mid = int(len(value) / 2)
                # mid=int(len(value)/2 + 1)
                #
                # left = value[:mid]#[-seq_len + 1:]
                #
                # arr = [-1 for i in range(seq_len - min_seq_len)] + \
                #       [v[0] for v in left]
                # # arr is [0,0...,0,itemid0,itemid1,....]
                # # each line user_id|item1,item2,...,item(self.lem-1)|label
                # for i in range(len(arr) - seq_len + 1):
                #     sample = arr[i: i + seq_len]
                #     f.write('{}|'.format(user))  # sample id
                #     f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                #     f.write("{}".format(sample[-1]))  # label, no ts
                #     f.write('\n')
        if False and validation_sample is not None:
            user_his_behav = _gen_user_his_behave(validation_sample)
            for user, value in user_his_behav.items():
                if len(value) / 2 + 1 < min_seq_len:
                    continue

                # mid = int(len(value) / 2)
                mid = int(len(value) / 2 + 1)

                left = value[:mid]  # [-seq_len + 1:]

                arr = [-1 for i in range(seq_len - min_seq_len)] + \
                      [v[0] for v in left]
                # arr is [0,0...,0,itemid0,itemid1,....]
                # each line user_id|item1,item2,...,item(self.lem-1)|label
                for i in range(len(arr) - seq_len + 1):
                    sample = arr[i: i + seq_len]
                    f.write('{}|'.format(user))  # sample id
                    f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                    f.write("{}".format(sample[-1]))  # label, no ts
                    f.write('\n')

    # Split train sample to segments
    _split_train_sample(train_instances_file,train_sample_seg_cnt)
    return stat
def _split_train_sample(train_instances_file,train_sample_seg_cnt=400):
    segment_filenames = []
    segment_files = []
    for i in range(train_sample_seg_cnt):
        filename = "{}_{}".format(train_instances_file, i)
        segment_filenames.append(filename)
        segment_files.append(open(filename, 'w'))

    with open(train_instances_file, 'r') as fi:
        for line in fi:
            i = random.randint(0, train_sample_seg_cnt - 1)# train_sample_seg_cnt is 400
            segment_files[i].write(line)

    for f in segment_files:
        f.close()

    os.remove(train_instances_file)

    # Shuffle
    num=0
    for fn in segment_filenames:
        lines = []
        with open(fn, 'r') as f:
            for line in f:
                lines.append(line)
        random.shuffle(lines)
        num+=len(lines)
        with open(fn, 'w') as f:
            for line in lines:
                f.write(line)
    print('number of training instance is {}'.format(num))

def _gen_test_sample(test_instances_file,seq_len=70,min_seq_len=6,user_his_behav=None):
    # user_his_behav is a dict, key is userid and value is a list [(itemId,timestamp)], list is ascendent by timestamp
    with open(test_instances_file, 'w') as f:
        for user, value in user_his_behav.items():
            value = user_his_behav[user]  # the clicked item id for user
            count = len(value)  # the item number of the user to click
            if count < min_seq_len:
                continue
            arr = [-1 for i in range(seq_len - min_seq_len)] + \
                  [v for v in value]
            for i in range(len(arr) - seq_len, len(arr) - seq_len + 1): # 
                sample = arr[i: i + seq_len]
                f.write('{}|'.format(user))  # sample id
                f.write("{}|".format(",".join([str(v) for v in sample[:-1]])))
                f.write("{}".format(sample[-1]))  # label, no ts
                f.write('\n')
                count += 1


