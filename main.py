import numpy as np
import gym
import tensorflow as tf
import pickle
import logz
import scipy.signal
import tf_util as U
from nosharing_cnn_policy import CnnPolicy
from mlp_policy import MlpPolicy
import dataset
from misc_util import zipsame, cg
from adam import Adam
from models import *
from simple import *
import replay_buffer
import build_graph


# def train(env, logz = None,  
#         min_timesteps_per_batch = 1024, # what to train on
#         max_kl = 0.01, 
#         cg_iters = 10,
#         gamma = 0.995, 
#         lam = 0.98, # advantage estimation
#         entcoeff=0.0,
#         cg_damping=1e-2,
#         vf_stepsize=3e-4,
#         vf_iters =3,
#         max_timesteps=0, max_episodes=0, max_iters=501,  # time constraint
#         callback=None
#         ):

#     sess = U.single_threaded_session()
#     sess.__enter__()
  
#     ob_dim = env.observation_space.shape[0]
#     num_actions = np.prod(env.action_space.shape)

#     ob_space = env.observation_space
#     ac_space = env.action_space


#     command = np.zeros(2)
#     cmd_space = len(command)    


#     # print "{}, {}".format(ob_dim, num_actions)

#     def policy_func(name, type, ob_space, ac_space): #pylint: disable=W0613
#         assert type in ['Cnn_Policy', 'Mlp_Policy']
#         if type == 'Cnn_Policy':
#             return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
#         elif type == 'Mlp_Policy':
#             return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
#         else:
#             print "Unknown Policy Type"

#     pi = policy_func(name = "pi", type = 'Cnn_Policy', ob_space = ob_space, ac_space = ac_space)
#     oldpi = policy_func(name = "oldpi", type = 'Cnn_Policy', ob_space = ob_space, ac_space = ac_space)

#     sy_ob = U.get_placeholder_cached(name="sy_ob")
#     sy_ac = pi.pdtype.sample_placeholder([None])

#     atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
#     ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

#     kloldnew = oldpi.pd.kl(pi.pd)
#     ent = pi.pd.entropy()
#     meankl = U.mean(kloldnew)
#     meanent = U.mean(ent)
#     entbonus = entcoeff * meanent

#     vferr = U.mean(tf.square(pi.vpred - ret))

#     ratio = tf.exp(pi.pd.logp(sy_ac) - oldpi.pd.logp(sy_ac)) # advantage * pnew / pold
#     surrgain = U.mean(ratio * atarg)

#     optimgain = surrgain + entbonus
#     losses = [optimgain, meankl, entbonus, surrgain, meanent]
#     loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

#     dist = meankl

#     all_var_list = pi.get_trainable_variables()
#     var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
#     vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]

#     vfadam = Adam(vf_var_list)

#     get_flat = U.GetFlat(var_list)
#     set_from_flat = U.SetFromFlat(var_list)
#     klgrads = tf.gradients(dist, var_list)
#     flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
#     shapes = [var.get_shape().as_list() for var in var_list]
#     start = 0
#     tangents = []
#     for shape in shapes:
#         sz = U.intprod(shape)
#         tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
#         start += sz
#     gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
#     fvp = U.flatgrad(gvp, var_list)

#     assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
#         for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
#     compute_losses = U.function([sy_ob, sy_ac, atarg], losses)
#     compute_lossandgrad = U.function([sy_ob, sy_ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
#     compute_fvp = U.function([flat_tangent, sy_ob, sy_ac, atarg], fvp)
#     compute_vfloss = U.function([sy_ob, ret], vferr)
#     compute_vflossandgrad = U.function([sy_ob, ret], U.flatgrad(vferr, vf_var_list))

#     U.initialize()
#     th_init = get_flat()

#     set_from_flat(th_init)

#     total_timesteps = 0
#     pickle_save = 10

#     for i in range(max_iters):
#         print("********** Iteration %i ************"%i)

#         # Collect paths until we have enough timesteps
#         timesteps_this_batch = 0
#         paths = []
#         while True:
#             ob = env.reset()
#             terminated = False

#             obs, acs, rewards, dones, vpreds = [], [], [], [], []
#             # animate_this_episode=(len(paths)==0 and (i % config.animation == 0) and animate)
#             while True:
#                 obs.append(ob)
#                 ac, vpred = pi.act(stochastic = True, sy_ob = ob)
#                 ac= ac.ravel()
#                 acs.append(ac)
#                 # vpred = vpred.ravel()
#                 vpreds.append(vpred)
#                 ob, rew, done, _ = env.step(ac)
#                 rewards.append(rew)
#                 dones.append(done)

#                 if done:
#                     break 

#             vpreds = np.concatenate(vpreds).ravel()
#             # print np.shape(vpreds.ravel())
#             path = {"observation" : np.array(obs), 
#                     "terminated" : terminated,
#                     "reward" : np.array(rewards), 
#                     "action" : np.array(acs),
#                     "done" : np.array(dones),
#                     "vpred" : np.array(vpreds)
#                     }
#                     # , 
#                     # "action_dists_mu": np.concatenate(action_dists_mu),
#                     # "action_dists_logstd": np.concatenate(action_dists_logstd)}

#             def pathlength(path):
#                 return len(path["reward"])
#             # print "KKKK"
#             # print np.shape(path["done"]), np.shape(path["vpred"]), np.shape(path["action"])
#             paths.append(path)
#             timesteps_this_batch += pathlength(path)
#             if timesteps_this_batch > min_timesteps_per_batch:
#                 break
#         total_timesteps += timesteps_this_batch
#         # Estimate advantage function
#         # vtargs, vpreds, advs = [], [], []
  
#         # print np.shape(obs)
#         # observ = np.concatenate([path["observation"] for path in paths])
#         # print np.shape(observ)

#         vtargs = []
#         for path in paths:
#             rew_t = path["reward"]
#             vpred_t = path["vpred"]
#             done_t = path["done"]

#             # print np.shape(rew_t), np.shape(vpred_t), np.shape(done_t)

#             T = len(rew_t)
#             path["adv"] = gaelam = np.empty(T, 'float32')
#             lastgaelam = 0
#             for t in reversed(range(T)):
#                 if done_t[t] == True:
#                     delta = rew_t[t] - vpred_t[t]
#                     gaelam[t] = lastgaelam = delta
#                 elif done_t[t] == False:
#                     delta = rew_t[t] + gamma * vpred_t[t+1] * (1-done_t[t]) - vpred_t[t]
#                     gaelam[t] = lastgaelam = delta + gamma * lam * (1-done_t[t]) * lastgaelam
#                 else:
#                     print "Error : Weire elements for calculating GAE"
#             # print "C1"
#             # print np.shape(path["adv"])
#             # print np.shape(path["vpred"])

#             path["tdlamret"] = path["adv"] + path["vpred"]
#             # print np.shape(path["tdlamret"])
#             # print "c1"
#             # print path["tdlamret"], len(path["tdlamret"])


#         # print len(paths)

#         ob_n = np.concatenate([path["observation"] for path in paths])
#         ac_n = np.concatenate([path["action"] for path in paths])
#         # ac_dist_mu_n = np.concatenate([path["action_dists_mu"] for path in paths])
#         # ac_dist_logstd_n = np.concatenate([path["action_dists_logstd"] for path in paths])
#         adv_n = np.concatenate([path["adv"] for path in paths])
#         tdlamret_n = np.concatenate([path["tdlamret"] for path in paths])
       
#         standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

#         # vtarg_n = np.concatenate(vtargs)
#         # vpred_n = np.concatenate(vpreds)

#         # print T
#         # print np.shape(ob_n)
#         # print np.shape(ac_n)
#         # print np.shape(tdlamret_n)

#         for _ in range(vf_iters):
#             for (mbob, mbret) in dataset.iterbatches((ob_n, tdlamret_n), 
#             include_final_partial_batch=False, batch_size=64):
#                 g = compute_vflossandgrad(mbob, mbret)
#                 vfadam.update(g, vf_stepsize)

#         # thprev = gf()

#         args = ob_n, ac_n, adv_n 
#         # seg["ob"], seg["ac"], seg["adv"]
#         fvpargs = [arr[::5] for arr in args]

#         def fisher_vector_product(p):
#             return compute_fvp(p, *fvpargs) + cg_damping * p

#         assign_old_eq_new() # set old parameter values to new parameter values
#         # feed = {sy_ob_no:ob_no, 
#         #         sy_ac_n:ac_n, 
#         #         sy_adv_n: standardized_adv_n}
#                 # , 
#                 # sy_oldlogits_mu_na: ac_dist_mu_n,
#                 # sy_oldlogits_logstd_na: ac_dist_logstd_n}

#         # def fisher_vector_product(p):
#         #     feed[flat_tangent] = p
#         #     return sess.run(fvp, feed) + cg_damping * p

#         surrbefore, _, _, _, _, g = compute_lossandgrad(*args)
#         # print surrbefore

#         # g = sess.run(pg, feed_dict = feed)
#         stepdir = U.conjugate_gradient(fisher_vector_product, g)

#         divide_iteration = 20

#         shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
#         lm = np.sqrt(shs / max_kl)
#         fullstep = stepdir / lm
#         expectedimprove = g.dot(fullstep)
#         # surrbefore = lossbefore[0]
#         stepsize = 1.0

#         thbefore = get_flat()

#         if np.allclose(g, 0):
#             print "Not Update, gradient zero"
#         else:       
#             for _ in range(10):
#                 thnew = thbefore + fullstep * stepsize
#                 set_from_flat(thnew)
#                 meanlosses = surr, kl, _, _, _ = np.array(compute_losses(*args))
#                 # optimgain, meankl, entbonus, surrgain, meanent
#                 improve = surr - surrbefore
#                 print "Expected: {0:0.3f} Actual: {0:0.3f}".format(expectedimprove, improve)
#                 if not np.isfinite(meanlosses).all():
#                     print "Got non-finite value of losses -- bad!"
#                 elif kl > max_kl * 1.5:
#                     print "violated KL constraint. shrinking step."
#                 elif improve < 0:
#                     print "surrogate didn't improve. shrinking step."
#                 else:
#                     print "Stepsize OK!" 
#                     break
#                 stepsize *= .5
#             else:
#                 print "couldn't compute a good step"
#                 set_from_flat(thbefore)

#         logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
#         logz.log_tabular("EpBest100RewMean", np.amax([path["reward"].sum() for path in paths]))
#         logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
#         logz.log_tabular("KLOldNew", kloldnew)
#         # logz.log_tabular("Entropy", entropy)
#         # logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
#         # logz.log_tabular("EVAfter", explained_variance_1d(vf_1.predict(ob_no), vtarg_n))
#         logz.log_tabular("TimestepsSoFar", total_timesteps)
#         # If you're overfitting, EVAfter will be way larger than EVBefore.
#         # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
#         logz.dump_tabular()

def wrap_train(env):
    from atari_wrappers import (wrap_deepmind, FrameStack)
    env = wrap_deepmind(env, episode_life = False, clip_rewards=False)
    env = FrameStack(env, 4)
    return env


def main():
    env = gym.make("PongNoFrameskip-v4")
    # Remove Scaled Float Frame wrapper, re-use if needed.
    env = wrap_train(env)
    model = cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )
    print "Come1"
    act = learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False
    )
    act.save("pong_model.pkl")
    env.close()


if __name__ == '__main__':
    main()



# def main():
#     # Setup Env
#     env = gym.make('Reacher-v1')
#     env = wrap_train(env)

#     # Setup Logger
#     logz.configure_output_dir(None)

#     num_timesteps = 40e6

#     # Start Train
#     train(env, logz, min_timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=1e-3,
#         max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=3, vf_stepsize=1e-4, entcoeff=0.00)

# if __name__ == "__main__":
#     main()


