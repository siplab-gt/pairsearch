import sys
import numpy as np
import simplejson
import utilsCrowdKernel as CKL   # local file
import scipy.io as sio
import time

if len(sys.argv) < 2:
    print('must supply dimension')
    sys.exit(1)

# load data
DATASET_HOME='./data'

dset = simplejson.load(open(DATASET_HOME+'/dataset.json'))
ftrs = np.load(DATASET_HOME+'/features.npy')

uuid_map = {uuid: i for i,uuid in enumerate(dset['image_uuids'])}
triplets = []
for line in open(DATASET_HOME+'/all-triplets.txt').readlines():
    (a,b,c) = line.replace('\n','').split(' ')
    triplets.append( (uuid_map[a], uuid_map[b], uuid_map[c]) )
triplets = np.array(triplets)

# embedding parameters
d = int(sys.argv[1])    # embedding dimension
n = len(uuid_map)       # number of points
m = triplets.shape[0]   # number of triplets

# the data set's triplets are of the form |t[0]-t[1]| < |t[0]-t[2]|
# make them like |q[0]-q[2]| < |q[1]-q[2]| as CKL expects
triplets = triplets[:,(1,2,0)]

subsampling = 0
if subsampling:
    np.random.shuffle(triplets)
    triplets = triplets[:int(m*0.1)]
    m = triplets.shape[0]

mtest = int(m * 0.1)
mtrain = m - mtest
I = np.random.permutation(m)

print('Train set = %d, Test set = %d' %(mtrain, mtest))

# compute embedding 
Strain = triplets[I[:mtrain]]
X,emp_loss_train = CKL.computeEmbedding(n, d, Strain, mu=.01,
        num_random_restarts=2, epsilon=0, verbose=True)

# compute loss on test set
Stest = triplets[I[mtrain:]]
emp_loss_test,hinge_loss_test,log_loss_test = CKL.getLoss(X,Stest)

print()
print('Training loss = %f,   Test loss = %f'
        % (emp_loss_train, emp_loss_test))

fname = ('output-d%d-' % d) + time.strftime("%Y%m%d-%H%M%S")
sio.savemat(DATASET_HOME + '/' + fname, {'X': X})

