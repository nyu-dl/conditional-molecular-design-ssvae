from __future__ import print_function

import numpy as np
import tensorflow as tf


class Model(object):

    def __init__(self, seqlen_x, dim_x, dim_y, dim_z=100, dim_h=250, n_hidden=3, batch_size=200, beta=10000., char_set=[' ']):

        self.seqlen_x, self.dim_x, self.dim_y, self.dim_z, self.dim_h, self.n_hidden, self.batch_size = seqlen_x, dim_x, dim_y, dim_z, dim_h, n_hidden, batch_size
        self.beta = beta
        
        self.char_to_int = dict((c,i) for i,c in enumerate(char_set))
        self.int_to_char = dict((i,c) for i,c in enumerate(char_set))
        
        self.G = tf.Graph()
        self.G.as_default()

        ## variables for labeled data
        self.x_L = tf.placeholder(tf.float32, [None, self.seqlen_x, self.dim_x])
        self.xs_L = tf.placeholder(tf.float32, [None, self.seqlen_x, self.dim_x])
        self.y_L = tf.placeholder(tf.float32, [None, self.dim_y])

        ## functions for labeled data
        self.classifier_L_out = self._rnnpredictor(self.x_L, self.dim_h, 2*self.dim_y, reuse = False)
        self.y_L_mu, self.y_L_lsgms = tf.split(self.classifier_L_out, [self.dim_y, self.dim_y], 1)
        self.y_L_sample = self._draw_sample(self.y_L_mu, self.y_L_lsgms)

        self.encoder_L_out = self._rnnencoder(self.x_L, self.y_L, self.dim_h, 2*self.dim_z, reuse = False)
        self.z_L_mu, self.z_L_lsgms = tf.split(self.encoder_L_out, [self.dim_z, self.dim_z], 1)
        self.z_L_sample = self._draw_sample(self.z_L_mu, self.z_L_lsgms)

        self.decoder_L_out = self._rnndecoder(self.xs_L, tf.concat([self.z_L_sample, self.y_L], 1), self.dim_h, self.dim_x, reuse = False)
        self.x_L_recon = tf.nn.softmax(self.decoder_L_out)

        self.decoder_DL_out = self._rnndecoder(self.xs_L, tf.concat([self.z_L_mu, self.y_L], 1), self.dim_h, self.dim_x, reuse = True)
        self.x_DL_recon = tf.nn.softmax(self.decoder_DL_out)

        self.z_G = tf.placeholder(tf.float32, [None, dim_z])
        self.decoder_G_out = self._rnndecoder(self.xs_L, tf.concat([self.z_G, self.y_L], 1), self.dim_h, self.dim_x, reuse = True)
        self.x_G_recon = tf.nn.softmax(self.decoder_G_out)


        ## variables for unlabeled data
        self.x_U = tf.placeholder(tf.float32, [None, self.seqlen_x, self.dim_x])
        self.xs_U = tf.placeholder(tf.float32, [None, self.seqlen_x, self.dim_x])

        ## functions for unlabeled data
        self.classifier_U_out = self._rnnpredictor(self.x_U, self.dim_h, 2*self.dim_y, reuse = True)
        self.y_U_mu, self.y_U_lsgms = tf.split(self.classifier_U_out, [self.dim_y, self.dim_y], 1)
        self.y_U_sample = self._draw_sample(self.y_U_mu, self.y_U_lsgms)

        self.encoder_U_out = self._rnnencoder(self.x_U, self.y_U_sample, self.dim_h, 2*self.dim_z, reuse = True)
        self.z_U_mu, self.z_U_lsgms = tf.split(self.encoder_U_out, [self.dim_z, self.dim_z], 1)
        self.z_U_sample = self._draw_sample(self.z_U_mu, self.z_U_lsgms)

        self.decoder_U_out = self._rnndecoder(self.xs_U, tf.concat([self.z_U_sample, self.y_U_sample], 1), self.dim_h, self.dim_x, reuse = True)
        self.x_U_recon = tf.nn.softmax(self.decoder_U_out)

        self.encoder_U2_out = self._rnnencoder(self.x_U, self.y_U_mu, self.dim_h, 2*self.dim_z, reuse = True)
        self.z_U2_mu, self.z_U2_lsgms = tf.split(self.encoder_U2_out, [self.dim_z, self.dim_z], 1)
        
        self.decoder_DU_out = self._rnndecoder(self.xs_U, tf.concat([self.z_U2_mu, self.y_U_mu], 1), self.dim_h, self.dim_x, reuse = True)
        self.x_DU_recon = tf.nn.softmax(self.decoder_DU_out)


        self.saver = tf.train.Saver()
        self.session = tf.Session()
        

    def train(self, trnX_L, trnXs_L, trnY_L, trnX_U, trnXs_U, valX_L, valXs_L, valY_L, valX_U, valXs_U):

        self.mu_prior=np.mean(trnY_L,0)   
        self.cov_prior=np.cov(trnY_L.T)     

        self.tf_mu_prior=tf.constant(self.mu_prior, shape=[1, self.dim_y], dtype=tf.float32)   
        self.tf_cov_prior=tf.constant(self.cov_prior, shape=[self.dim_y, self.dim_y], dtype=tf.float32)


        # objective functions
        objL = self._obj_L()
        objU = self._obj_U()
        objYpred_MSE = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.y_L, self.y_L_mu), 1))
        
        objL_val = - tf.reduce_mean(- tf.reduce_sum(self.cross_entropy(tf.layers.flatten(self.x_L), tf.layers.flatten(self.x_DL_recon)), 1))
        objU_val = - tf.reduce_mean(- tf.reduce_sum(self.cross_entropy(tf.layers.flatten(self.x_U), tf.layers.flatten(self.x_DU_recon)), 1))

        batch_size_L=int(self.batch_size*len(trnX_L)/(len(trnX_L)+len(trnX_U)))
        batch_size_U=int(self.batch_size*len(trnX_U)/(len(trnX_L)+len(trnX_U)))
        n_batch=int(len(trnX_L)/batch_size_L)
        
        batch_size_val_L=int(len(valX_L)/10)
        batch_size_val_U=int(len(valX_U)/10)

        cost = (objL * float(batch_size_L) + objU * float(batch_size_U))/float(batch_size_L+batch_size_U) + float(batch_size_L)/float(batch_size_L+batch_size_U) * (self.beta * objYpred_MSE)
        cost_val = objYpred_MSE
        train_op = tf.train.AdamOptimizer().minimize(cost)
        self.session.run(tf.global_variables_initializer())


        # training
        val_log=np.zeros(300)
        for epoch in range(300):
            [trnX_L, trnXs_L, trnY_L]=self._permutation([trnX_L, trnXs_L, trnY_L])
            [trnX_U, trnXs_U]=self._permutation([trnX_U, trnXs_U])

            for i in range(n_batch):
                start_L=i*batch_size_L
                end_L=start_L+batch_size_L
                
                start_U=i*batch_size_U
                end_U=start_U+batch_size_U

                trn_res = self.session.run([train_op, cost, objL, objU, objYpred_MSE],
                                      feed_dict = {self.x_L: trnX_L[start_L:end_L], self.xs_L: trnXs_L[start_L:end_L], self.y_L: trnY_L[start_L:end_L],
                                      self.x_U: trnX_U[start_U:end_U], self.xs_U: trnXs_U[start_U:end_U]})

            val_res = []
            for i in range(10):
                start_L=i*batch_size_val_L
                end_L=start_L+batch_size_val_L
                
                start_U=i*batch_size_val_U
                end_U=start_U+batch_size_val_U
            
                val_res.append(self.session.run([cost_val, objL_val, objU_val, objYpred_MSE],
                                  feed_dict = {self.x_L: valX_L[start_L:end_L], self.xs_L: valXs_L[start_L:end_L], self.y_L: valY_L[start_L:end_L],
                                  self.x_U: valX_U[start_U:end_U], self.xs_U: valXs_U[start_U:end_U]}))
            
            val_res=np.mean(val_res,axis=0)
            print(epoch, ['Training', 'cost_trn', trn_res[1]])
            print('---', ['Validation', 'cost_val', val_res[0]])

            val_log[epoch] = val_res[0]
            if epoch > 20 and np.min(val_log[0:epoch-10]) * 0.99 < np.min(val_log[epoch-10:epoch+1]):
                print('---termination condition is met')
                break


    def predict(self, x_input):

        return self.session.run(self.y_U_mu, feed_dict = {self.x_U: x_input})


    def latent(self, x_input, y_input):

        return self.session.run(self.z_L_mu, feed_dict = {self.x_L: x_input, self.y_L: y_input})


    def sampling_unconditional(self): 
       
        sample_z=np.random.randn(1, self.dim_z)
        sample_y=np.random.multivariate_normal(self.mu_prior, self.cov_prior, 1)      
          
        sample_smiles=self.beam_search(sample_z, sample_y, k=5)

        return sample_smiles
        
    
    def sampling_conditional(self, yid, ytarget):
    
        def random_cond_normal(yid, ytarget):

            id2=[yid]
            id1=np.setdiff1d([0,1,2],id2)
        
            mu1=self.mu_prior[id1]
            mu2=self.mu_prior[id2]
            
            cov11=self.cov_prior[id1][:,id1]
            cov12=self.cov_prior[id1][:,id2]
            cov22=self.cov_prior[id2][:,id2]
            cov21=self.cov_prior[id2][:,id1]
            
            cond_mu=np.transpose(mu1.T+np.matmul(cov12, np.linalg.inv(cov22)) * (ytarget-mu2))[0]
            cond_cov=cov11 - np.matmul(np.matmul(cov12, np.linalg.inv(cov22)), cov21)
            
            marginal_sampled=np.random.multivariate_normal(cond_mu, cond_cov, 1)
            
            tst=np.zeros(3)
            tst[id1]=marginal_sampled
            tst[id2]=ytarget
            
            return np.asarray([tst])

        sample_z=np.random.randn(1, self.dim_z)
        sample_y=random_cond_normal(yid, ytarget) 
          
        sample_smiles=self.beam_search(sample_z, sample_y, k=5)
            
        return sample_smiles


    def beam_search(self, z_input, y_input, k=5):

        def reconstruct(xs_input, z_sample, y_input):

            return self.session.run(self.x_G_recon, feed_dict = {self.xs_L: xs_input, self.z_G: z_sample, self.y_L: y_input})
        
        
        cands=np.asarray([np.zeros((1, self.seqlen_x, self.dim_x), dtype=np.float32)] )
        cands_score=np.asarray([100.])
        
        for i in range(self.seqlen_x-1):
        
            cands2=[]
            cands2_score=[]

            for j, samplevec in enumerate(cands):
                o = reconstruct(samplevec, z_input, y_input)
                sampleidxs = np.argsort(-o[0,i])[:k]
                
                for sampleidx in sampleidxs: 
                    
                    samplevectt=np.copy(samplevec)
                    samplevectt[0, i+1, sampleidx] = 1.
                    
                    cands2.append(samplevectt)
                    cands2_score.append(cands_score[j] * o[0,i,sampleidx])
                    
            cands2_score=np.asarray(cands2_score)
            cands2=np.asarray(cands2)
            
            kbestid = np.argsort(-cands2_score)[:k]
            cands=np.copy(cands2[kbestid])
            cands_score=np.copy(cands2_score[kbestid])
            
            if np.sum([np.argmax(c[0][i+1]) for c in cands])==0:
                break

        sampletxt = ''.join([self.int_to_char[np.argmax(t)] for t in cands[0,0]]).strip()

        return sampletxt


    def _obj_L(self):

        L_log_lik = - tf.reduce_sum(self.cross_entropy(tf.layers.flatten(self.x_L), tf.layers.flatten(self.x_L_recon)), 1)
        L_log_prior_y = self.noniso_logpdf(self.y_L)
        L_KLD_z = self.iso_KLD(self.z_L_mu, self.z_L_lsgms)

        objL = - tf.reduce_mean(L_log_lik + L_log_prior_y - L_KLD_z)
        
        return objL


    def _obj_U(self):

        U_log_lik = - tf.reduce_sum(self.cross_entropy(tf.layers.flatten(self.x_U), tf.layers.flatten(self.x_U_recon)), 1)
        U_KLD_y = self.noniso_KLD(self.y_U_mu, self.y_U_lsgms)
        U_KLD_z = self.iso_KLD(self.z_U_mu, self.z_U_lsgms)

        objU = - tf.reduce_mean(U_log_lik - U_KLD_y - U_KLD_z)
        
        return objU


    def cross_entropy(self, x, y, const = 1e-10):
        return - ( x*tf.log(tf.clip_by_value(y, const, 1.0))+(1.0-x)*tf.log(tf.clip_by_value(1.0-y, const, 1.0)) )


    def iso_KLD(self, mu, log_sigma_sq):
        return tf.reduce_sum( - 0.5 * (1.0 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq) ), 1)


    def noniso_logpdf(self, x):
        return - 0.5 * (float(self.cov_prior.shape[0]) * np.log(2.*np.pi) +  np.log(np.linalg.det(self.cov_prior))
                        + tf.reduce_sum( tf.multiply( tf.matmul( tf.subtract(x, self.tf_mu_prior), tf.matrix_inverse(self.tf_cov_prior) ), tf.subtract(x, self.tf_mu_prior) ), 1) )


    def noniso_KLD(self, mu, log_sigma_sq):
        return 0.5 * ( tf.trace( tf.scan(lambda a, x: tf.matmul(tf.matrix_inverse(self.tf_cov_prior), x), tf.matrix_diag(tf.exp(log_sigma_sq)) ) ) 
                      + tf.reduce_sum( tf.multiply( tf.matmul( tf.subtract(self.tf_mu_prior, mu), tf.matrix_inverse(self.tf_cov_prior) ), tf.subtract(self.tf_mu_prior, mu) ), 1)
                      - float(self.cov_prior.shape[0]) + np.log(np.linalg.det(self.cov_prior)) - tf.reduce_sum(log_sigma_sq, 1) )  


    def _permutation(self, set):

        permid=np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i]=set[i][permid]

        return set


    def _draw_sample(self, mu, lsgms):

        epsilon = tf.random_normal((tf.shape(mu)), 0, 1)
        sample = tf.add(mu, tf.multiply(tf.exp(0.5*lsgms), epsilon))

        return sample 


    def _rnnpredictor(self, x, dim_h, dim_y, reuse=False):

        with tf.variable_scope('rnnpredictor', reuse=reuse):

            cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(dim_h) for _ in range(self.n_hidden)])
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(dim_h) for _ in range(self.n_hidden)])
            init_state_fw = cell_fw.zero_state(tf.shape(x)[0], tf.float32)
            init_state_bw = cell_bw.zero_state(tf.shape(x)[0], tf.float32)
            
            _, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
            res = tf.layers.dense(tf.concat([final_state[0][-1],final_state[1][-1]], 1), dim_y)
            
            
        return res


    def _rnnencoder(self, x, st, dim_h, dim_y, reuse=False):

        with tf.variable_scope('rnnencoder', reuse=reuse):

            cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(dim_h) for _ in range(self.n_hidden)])
            cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(dim_h) for _ in range(self.n_hidden)])
            init_state_fw = tf.layers.dense(st, dim_h, activation = tf.nn.sigmoid)
            init_state_bw = tf.layers.dense(st, dim_h, activation = tf.nn.sigmoid)
            peek_in = tf.layers.dense(st, self.dim_x, activation = tf.nn.sigmoid)
            peek = tf.reshape(tf.tile(peek_in, [1, self.seqlen_x]), [-1, self.seqlen_x, self.dim_x])
            
            _, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, tf.concat([x,peek],2),
                                initial_state_fw=tuple([init_state_fw]*self.n_hidden), initial_state_bw=tuple([init_state_bw]*self.n_hidden))
            res = tf.layers.dense(tf.concat([final_state[0][-1],final_state[1][-1]], 1), dim_y)
            
            
        return res


    def _rnndecoder(self, x, st, dim_h, dim_y, reuse=False):

        with tf.variable_scope('rnndecoder', reuse=reuse):
        
            cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(dim_h) for _ in range(self.n_hidden)])
            init_state = tf.layers.dense(st, dim_h, activation = tf.nn.sigmoid)
            peek_in = tf.layers.dense(st, self.dim_x, activation = tf.nn.sigmoid)
            peek = tf.reshape(tf.tile(peek_in, [1, self.seqlen_x]), [-1, self.seqlen_x, self.dim_x])

            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, tf.concat([x,peek],2), initial_state=tuple([init_state]*self.n_hidden))
            res = tf.layers.dense(rnn_outputs, dim_y)


        return res
