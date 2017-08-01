# Tensorflow Implementation of "A Distributional Perspectives on Reinforcement Learning"

Please refer to the original paper by [Marc G. Bellemare, Will Dabney, and Remi Munos](https://arxiv.org/abs/1707.06887) for more detail.

[Environment]
 * python 2.7
 * Tensorflow 1.2.1
 * [OpenAI Gym](https://gym.openai.com/) and [PongNoFrameskip-v4](https://gym.openai.com/envs/Pong-v0): I used NoFrameSkip and version.

[Acknowledgement]
 * I implemented this one based on openAI's [baselines code DQN](https://github.com/openai/baselines) with some minor changes to use on python 2.7

[Network Structure]
 * Mostly same with the original DQN, except for the changes in output. Number of output is increased from [# of actions] to [# of actions x # of atoms] to express the distribution of value.
 * After getting above outputs, we need to make it as probability using softmax for each action.

[Training]
 * run : python main.py
 * for PongNoFrameskip-v4: performances starts to increase after 350k steps.
 * for Atari, BreakoutNoFrameskip-v4: there are binary outputs, one converges to around 5, the other to over 300.
 * For rest of environments, I couldn't test it everything. So, please run it and let me know the results.

[Results]
 * Will be updated soon.

[Important code for implementation]

 * Major changes are in build_graph.py 

```python
build_dist_train(...)
{
	...
    
        # (4) Make T_z, b_j, l, u in matrix form

        z = tf.tile(tf.reshape(tf.range(-V_max, V_max + delta_z, delta_z), [1, num_atoms]), [batch_size, 1])
        r = tf.tile(tf.reshape(rew_t_ph, [batch_size, 1]), [1, num_atoms])

        done = tf.tile(tf.reshape(done_mask_ph, [batch_size, 1]), [1, num_atoms])
        
        T_z = r + z * gamma * (1 - done)
        T_z = tf.maximum(tf.minimum(T_z, V_max), V_min) # Restrict upper and lower value of T_z to V_max and V_min
        b = (T_z - V_min) / delta_z
        l, u = tf.floor(b), tf.ceil(b)
        l_id = tf.cast(l, tf.int32)
        u_id = tf.cast(u, tf.int32)

        v_dist_t_selected = tf.reshape(v_dist_t_selected, [-1])
        add_index = tf.range(batch_size) * num_atoms

        err = tf.zeros([batch_size])

        for j in range(num_atoms):
            l_index = l_id[:, j] + add_index
            u_index = u_id[:, j] + add_index

            p_tl = tf.gather(v_dist_t_selected, l_index)
            p_tu = tf.gather(v_dist_t_selected, u_index)
            log_p_tl = tf.log(p_tl)
            log_p_tu = tf.log(p_tu)
            p_tp1 = v_dist_tp1_selected[:,j]
            err = err + p_tp1 * ((u[:,j] - b[:,j]) * log_p_tl + (b[:,j] - l[:,j]) * log_p_tu)  	
});
```
