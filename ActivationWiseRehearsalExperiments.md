<center><b>Experiment #1</b></center>
<center>Getting proof of concept</center> 

**General idea**

Amon Elders
We can augment this by keeping the individual moments in memory and try out activation-wise rehearsal from a normal distribution and just see what happens. This would be besides keeping track of the moving moments and training with those. We can do one, or the other, or both. (edited)
Like a normal distribution may just capture the distribution well-enough for this to act as an appropriate shift when doing rehearsal.

It's something like saying 'let's assume our memory buffer is normally distributed', and we just keep track of the means/variances of the minibatches we have seen.

So instead of keeping track of a memory buffer of samples from minibatches that we have seen. We keep track of the means/variances of the mini-batches and sample from those activation-wise using a normal distribution. (edited) 

(which should act as a compression of the mini-buffer)
(and we may find surprising things by doing activation-wise rehearsal instead of the normal rehearsal)

It's like modeling the outputs of the activations of every layer as a normal distribution.

**Dataset:**  
MNIST data set (torch vision as source for it)

**Models**  
- One simple 5 layer MLP which keeps track of the batch norm statistics of the minibatches that it has seen. (we don't apply batch norm tho!)

That is. We have a memory that keeps track of the minibatch statistics at every layer:

Minibatch 1: u^{1}_{1st layer}, \sigma^{1}_{1nd layer}, u^{1}_{2nd layer}, \sigma^{1}_{2nd layer}, ...... , u^{1}_{5th layer}, \sigma^{1}_{5th layer} 
Minibatch 2: u^{2}_{1st layer}, \sigma^{2}_{1nd layer}, u^{2}_{2nd layer}, \sigma^{2}_{2nd layer}, ...... , u^{2}_{5th layer}, \sigma^{2}_{5th layer} 

And, when we receive Minibatch 2, we rehearse -activation wise- from a -normal distribution- using the moments u^{1}, and \sigma^{1} that we kept in memory. 
So that we train on Minibatch_adjusted 2, instead of on Minibatch 2.

When we receive Minibatch 3, we can rehearse -activiation wise- from a normal distribution using both the moments u^{1}, and sigma{1}, u^{2}, and sigma{2}.

Or something else we can try is to keep track of the moments of Minibatch_adjusted2, and rehearse from those u^{2_adjusted}, sigma^{2_adjusted} instead of from both u^{1},u^{2}, sigma^{1}/sigma^{2}

Hyperparameter: r_{1} = n_{rehearse_{1}}/n_{minibatch} --> The ratio of how much we rehearse per minibatch seen versus the cardinality of the minibatch that comes in.
r_{2} = n_{rehearse_{2}}/n_{minibatch}

**Experiment setup**  
Stream in the data - in batches - with two classes at the same time.

That is, we feed the network mini-batches of zeros and ones (0,1), and train on it
Then, we feed the network mini-batches of twos and threes (2,3), until the (0,1)'s are forgotten. 

Now we know how much twos and threes we need to feed to the nethwork to forget the (0,1)'s. Then we start over.

We feed the network the mini-batches of zeros and ones (0,1), and train on it. Then, we feed the network the amount of mini-batches of twos and threes such that we know it will forget the (0,1)'s 
**and** we apply our activation wise rehearsal adjustment.

Check. Do we still forget the (0,1)'s?

**Expectations**
Do we still forget the (0,1)'s after applying our activation wise rehearsal from a normal distribution using the batch norm statistics?

<center><b>Experiment #2</b></center>
<center>A more realistic CL scenario</center> 


**Dataset:**  
MNIST data set (torch vision as source for it)

**Models**  
- One simple 5 layer MLP which keeps track of the batch norm statistics of the minibatches that it has seen. (we don't apply batch norm tho!)

That is. We have a memory that keeps track of the minibatch statistics at every layer:

Minibatch 1: u^{1}_{1st layer}, \sigma^{1}_{1nd layer}, u^{1}_{2nd layer}, \sigma^{1}_{2nd layer}, ...... , u^{1}_{5th layer}, \sigma^{1}_{5th layer} 
Minibatch 2: u^{2}_{1st layer}, \sigma^{2}_{1nd layer}, u^{2}_{2nd layer}, \sigma^{2}_{2nd layer}, ...... , u^{2}_{5th layer}, \sigma^{2}_{5th layer} 

And, when we receive Minibatch 2, we rehearse -activation wise- from a -normal distribution- using the moments u^{1}, and \sigma^{1} that we kept in memory. 
So that we train on Minibatch_adjusted 2, instead of on Minibatch 2.

When we receive Minibatch 3, we can rehearse -activiation wise- from a normal distribution using both the moments u^{1}, and sigma{1}, u^{2}, and sigma{2}.

Or something else we can try is to keep track of the moments of Minibatch_adjusted2, and rehearse from those u^{2_adjusted}, sigma^{2_adjusted} instead of from both u^{1},u^{2}, sigma^{1}/sigma^{2}

Hyperparameter: r_{1} = n_{rehearse_{1}}/n_{minibatch} --> The ratio of how much we rehearse per minibatch seen versus the cardinality of the minibatch that comes in.
r_{2} = n_{rehearse_{2}}/n_{minibatch}

**Experiment setup**  


Stream in the data - in batches - with two classes at the same time.

That is, we feed the network mini-batches of zeros and ones (0,1), and train on it
Then, we feed the network mini-batches of twos and threes (2,3), apply our rehearsal adjustment, train on it
Then, we feed the network mini-batches of threes and fours (3,4), apply our rehearsal adjustment and train on it

**Expectations**
What happens to performance on all tasks over time? 

Need to plot prediction errors with respect to task specific test sets (so we need a small test set of (0,1)) and a small test set of (2,3)'s. And then as we loop over all of MNIST, does performance degrade/stay constant over all the tasks?




