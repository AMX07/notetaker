# Building makemore Part 5: Building a WaveNet

Hi everyone, today we are continuing our implementation of makemore, our favorite character-level language model.

Now you'll notice that the background behind me is different — that's because I am in Kyoto and it is awesome. So I'm in a hotel room here.

Now, over the last few lectures we've built up to this architecture that is a multi-layer perceptron character-level language model. So we see that it receives three previous characters and tries to predict the fourth character in a sequence using a very simple multi-layer perceptron using one hidden layer of neurons with tanh nonlinearities.

So what we'd like to do now in this lecture is I'd like to complexify this architecture. In particular, we would like to take more characters in a sequence as an input — not just three — and in addition to that, we don't just want to feed them all into a single hidden layer because that squashes too much information too quickly. Instead, we would like to make a deeper model that progressively fuses this information to make its guess about the next character in a sequence.

And so we'll see that as we make this architecture more complex, we're actually going to arrive at something that looks very much like a WaveNet. The WaveNet is this paper published by DeepMind in 2016 and it is also a language model basically, but it tries to predict audio sequences instead of character-level sequences or word-level sequences. But fundamentally, the modeling setup is identical — it is an autoregressive model and it tries to predict the next character in a sequence. And the architecture actually takes this interesting hierarchical sort of approach to predicting the next character in a sequence with a tree-like structure, and this is the architecture we're going to implement in the course of this video.

## Starter Code Walkthrough

So let's get started. The starter code for Part 5 is very similar to where we ended up in Part 3. Recall that Part 4 was the manual backpropagation exercise — that is kind of an aside. So we are coming back to Part 3, copy-pasting chunks out of it, and that is our starter code for Part 5. I've changed very few things otherwise, so a lot of this should look familiar if you've gone through Part 3.

So in particular, very briefly, we are doing imports, we are reading our dataset of words, and we are processing our set of words into individual examples. None of this data generation code has changed, and basically we have lots and lots of examples. In particular, we have 182,000 examples of three characters trying to predict the fourth one, and we've broken up every one of these words into little problems of "given three characters, predict the fourth one." So this is our dataset and this is what we're trying to get the neural net to do.

Now, in Part 3 we started to develop our code around these layer modules that are, for example, like `class Linear`, and we're doing this because we want to think of these modules as building blocks — like Lego building block bricks that we can sort of stack up into neural networks. We can feed data between these layers and stack them up into sort of graphs.

Now, we also developed these layers to have APIs and signatures very similar to those that are found in PyTorch. So we have `torch.nn` and it's got all these layer building blocks that you would use in practice, and we were developing all of these to mimic the APIs of those. So for example, we have `Linear`, so there will also be a `torch.nn.Linear`, and its signature will be very similar to our signature, and the functionality will be also quite identical as far as I'm aware.

So we have the `Linear` layer, the `BatchNorm1d` layer, and the `Tanh` layer that we developed previously. `Linear` is just a matrix multiply in the forward pass of this module. `BatchNorm` of course is this crazy layer that we developed in the previous lecture, and what's crazy about it is — well, there's many things. Number one, it has these running mean and variances that are trained outside of backpropagation; they are trained using exponential moving average inside this layer when we call the forward pass.

In addition to that, there's this training flag because the behavior of BatchNorm is different during train time and evaluation time, and so suddenly we have to be very careful that BatchNorm is in its correct state — that it's in the evaluation state or training state. So that's something to now keep track of, something that sometimes introduces bugs because you forget to put it into the right mode.

And finally, we saw that BatchNorm couples the statistics — or the activations — across the examples in the batch. So normally we thought of the batch as just an efficiency thing, but now we are coupling the computation across batch elements, and it's done for the purposes of controlling the activation statistics as we saw in the previous video.

So it's a very weird layer, causes a lot of bugs — partly, for example, because you have to modulate the training and eval phase, and so on. In addition, for example, you have to wait for the mean and the variance to settle and to actually reach a steady state, and so you have to make sure that — basically, there's state in this layer and state is harmful usually.

Now, I brought out the generator object. Previously we had a `generator = g` and so on inside these layers. I've discarded that in favor of just initializing the torch RNG outside here — use it just once globally, just for simplicity.

And then here we are starting to build out some of the neural network elements. This should look very familiar — we have our embedding table `C`, and then we have a list of layers: a `Linear` feeds to `BatchNorm` feeds to `Tanh`, and then a linear output layer, and its weights are scaled down so we are not confidently wrong at the initialization. We see that this is about 12,000 parameters. We're telling PyTorch that the parameters require gradients.

The optimization is, as far as I'm aware, identical and should look very, very familiar. Nothing changed here.

The loss function looks very crazy — we should probably fix this, and that's because 32 batch elements are too few, and so you can get very lucky or unlucky in any one of these batches, and it creates a very thick loss function. So we're going to fix that soon.

Now, once we want to evaluate the trained neural network, we need to remember — because of the BatchNorm layers — to set all the layers to be `training = False`. So this only matters for the BatchNorm layer so far. And then we evaluate.

We see that currently we have a validation loss of 2.10, which is fairly good, but there's still ways to go. But even at 2.10, we see that when we sample from the model we actually get relatively name-like results that do not exist in a training set — so, for example, Yvonne, Kilo, Pros, Alaia, etc. So certainly not unreasonable I would say, but not amazing, and we can still push this validation loss even lower and get much better samples that are even more name-like.

So let's improve this model.

## Let's Fix the Learning Rate Plot

Okay, first let's fix this graph because it is daggers in my eyes and I just can't take it anymore.

So `lossi`, if you recall, is a Python list of floats. So for example, the first 10 elements. Now, what we'd like to do basically is we need to average up some of these values to get a more sort of representative value along the way.

So one way to do this is the following: in PyTorch, if I create for example a tensor of the first 10 numbers, then this is currently a one-dimensional array. But recall that I can view this array as two-dimensional — so for example, I can view it as a 2 by 5 array, and this is a 2D tensor now, 2 by 5. And you see what PyTorch has done is that the first row of this tensor is the first five elements and the second row is the second five elements.

I can also view it as a 5 by 2 as an example. And then recall that I can also use negative one in place of one of these numbers, and PyTorch will calculate what that number must be in order to make the number of elements work out. So this can be this or like that, but it will work — of course this would not work.

Okay, so this allows it to spread out some of the consecutive values into rows, so that's very helpful. Because what we can do now is: first of all, we're going to create a `torch.tensor` out of the `lossi` list of floats, and then we're going to view it as whatever it is but we're going to stretch it out into rows of 1,000 consecutive elements. So the shape of this now becomes 200 by 1,000, and each row is one thousand consecutive elements in this list.

So that's very helpful because now we can do a mean along the rows, and the shape of this will just be 200. And so we've taken basically the mean on every row, so `plt.plot` of that should be something nicer — much better.

So we see that we basically made a lot of progress and then here, this is the learning rate decay. So here we see that the learning rate decay subtracted a ton of energy out of the system and allowed us to settle into sort of the local minimum in this optimization.

So this is a much nicer plot. Let me come up and delete the monster, and we're going to be using this going forward.

## PyTorchifying Our Code: Layers, Containers, torch.nn, and Fun Bugs

Now, next up, what I'm bothered by is that our forward pass is a little bit gnarly and takes way too many lines of code. So in particular, we see that we've organized some of the layers inside the layers list but not all of them, for no reason. So in particular, we see that we still have the embedding table as a special case outside of the layers, and in addition to that, the viewing operation here is also outside of our layers. So let's create layers for these and then we can add those layers to just our list.

So in particular, the two things that we need: here we have this embedding table and we are indexing at the integers inside the tensor `xb`. So that's an embedding table lookup just done with indexing. And then here we see that we have this view operation which, if you recall from the previous video, simply rearranges the character embeddings and stretches them out into a row. And effectively what that does is the concatenation operation, basically, except it's free because viewing is very cheap in PyTorch — no memory is being copied, we're just re-representing how we view that tensor.

So let's create modules for both of these operations: the embedding operation and the flattening operation.

So I actually wrote the code in just to save some time. We have a module `Embedding` and a module `Flatten`, and both of them simply do the indexing operation in the forward pass and the flattening operation here. And this `C` now will just become a `self.weight` inside an `Embedding` module.

And I'm calling these layers specifically `Embedding` and `Flatten` because it turns out that both of them actually exist in PyTorch. So in PyTorch we have `nn.Embedding` and it also takes the number of embeddings and the dimensionality of the embedding, just like we have here. But in addition, PyTorch takes in a lot of other keyword arguments that we are not using for our purposes yet.

And for `Flatten`, that also exists in PyTorch and it also takes additional keyword arguments that we are not using. So we have a very simple `Flatten`, but both of them exist in PyTorch — they're just a bit more complex.

And now that we have these, we can simply take out some of these special-cased things. So instead of `C`, we're just going to have an `Embedding` of `vocab_size` and `n_embd`, and then after the embedding we are going to `Flatten`. So let's construct those modules, and now I can take out the special case because now `C` is the embedding's weight and it's inside layers.

So this should just work. And then here our forward pass simplifies substantially because we don't need to do these outside of the layers explicitly — they're now inside layers. So we can delete those.

But now, to kick things off we want this little `x` which in the beginning is just `xb` — the tensor of integers specifying the identities of these characters at the input. And so these characters can now directly feed into the first layer and this should just work.

So let me come here and insert a break because I just want to make sure that the first iteration of this runs and then there's no mistake. So that ran properly, and basically we substantially simplified the forward pass here.

Okay, I'm sorry I changed my microphone, so hopefully the audio is a little bit better.

Now, one more thing that I would like to do in order to PyTorchify our code even further is that right now we are maintaining all of our modules in a naked list of layers, and we can also simplify this because we can introduce the concept of PyTorch containers. So in `torch.nn`, which we are basically rebuilding from scratch here, there's a concept of containers, and these containers are basically a way of organizing layers into lists or dicts and so on.

So in particular, there's a `Sequential` which maintains a list of layers and is a module class in PyTorch, and it basically just passes a given input through all the layers sequentially — exactly as we are doing here.

So let's write our own `Sequential`. I've written the code here, and basically the code for `Sequential` is quite straightforward: we pass in a list of layers which we keep here, and then given any input in a forward pass, we just call all the layers sequentially and return the result. In terms of the parameters, it's just all the parameters of the child modules.

So we can run this and we can again simplify this substantially, because we don't maintain this naked list of layers — we now have a notion of a model which is a module, and in particular is a `Sequential` of all these layers. And now parameters are simply just `model.parameters()`, and so that list comprehension now lives there.

And then here, the code again simplifies substantially because we don't have to do this forwarding here. Instead, we just call the model on the input data, and the input data here are the integers inside `xb`. So we can simply do `logits`, which are the outputs of our model, are simply the model called on `xb`. And then the cross entropy here takes the logits and the targets. So this simplifies substantially.

And then this looks good, so let's just make sure this runs — that looks good.

Now, here we actually have some work to do still, but I'm going to come back later. For now, there's no more `layers` — there's `model.layers`, but it's not easy to access attributes of these classes directly. So we'll come back and fix this later.

And then here, of course, this simplifies substantially as well because logits are the model called on `x`. And then these logits come here.

So we can evaluate the train and validation loss, which currently is terrible because we just initialized the neural net. And then we can also sample from the model, and this simplifies dramatically as well, because we just want to call the model on the context and out come logits. And these logits go into softmax and get the probabilities, etc.

So we can sample from this model.

What did I screw up? Okay, so I fixed the issue and we now get the result that we expect, which is gibberish, because the model is not trained because we re-initialized it from scratch.

The problem was that when I fixed this cell to be `model.layers` instead of just `layers`, I did not actually run the cell, and so our neural net was in training mode. And what caused the issue here is the BatchNorm layer, as BatchNorm layer is of the likes to do. Because BatchNorm was in training mode, and here we are passing in an input which is a batch of just a single example made up of the context.

And so if you are trying to pass in a single example into a BatchNorm that is in the training mode, you're going to end up estimating the variance using the input, and the variance of a single number is not a number because it is a measure of spread. So for example, the variance of just the single number five, you can see, is not a number, and so that's what happened in the BatchNorm — basically caused an issue, and then that polluted all of the further processing.

So all that we had to do was make sure that this runs, and we basically made the issue of — again, we didn't actually see the issue with the loss, we could have evaluated the loss, but we got the wrong result because BatchNorm was in the training mode. And so we still get a result, it's just the wrong result, because it's using the sample statistics of the batch whereas we want to use the running mean and running variance inside the BatchNorm.

And so, again, an example of introducing a bug inline because we did not properly maintain the state of what is training or not.

## Implementing WaveNet

### Overview: WaveNet

Okay, so I've rewritten everything and here's where we are. As a reminder, we have the training loss of 2.05 and validation 2.10.

Now, because these losses are very similar to each other, we have a sense that we are not overfitting too much on this task, and we can make additional progress in our performance by scaling up the size of the neural network and making everything bigger and deeper.

Now, currently we are using this architecture here where we are taking in some number of characters going into a single hidden layer and then going to the prediction of the next character. The problem here is we don't have a naive way of making this bigger in a productive way. We could of course use our layers sort of building blocks and materials to introduce additional layers here and make the network deeper, but it is still the case that we are crushing all of the characters into a single layer all the way at the beginning. And even if we make this a bigger layer and add neurons, it's still kind of silly to squash all that information so fast in a single step.

So what we'd like to do instead is we'd like our network to look a lot more like this — in the WaveNet case. So you see in the WaveNet, when we are trying to make the prediction for the next character in the sequence, it is a function of the previous characters that feed in. But all of these different characters are not just crushed to a single layer and then you have a sandwich — they are crushed slowly.

So in particular, we take two characters and we fuse them into sort of like a bigram representation, and we do that for all these characters consecutively. And then we take the bigrams and we fuse those into four-character-level chunks, and then we fuse that again. And so we do that in this tree-like hierarchical manner — we fuse the information from the previous context slowly into the network as it gets deeper. And so this is the kind of architecture that we want to implement.

Now, in the WaveNet's case, this is a visualization of a stack of dilated causal convolution layers, and this makes it sound very scary, but actually the idea is very simple. The fact that it's a dilated causal convolution layer is really just an implementation detail to make everything fast. We're going to see that later, but for now let's just keep the basic idea of it, which is this progressive fusion. We want to make the network deeper, and at each level we want to fuse only two consecutive elements: two characters, then two bigrams, then two four-grams, and so on.

### Bumping the Context Size to 8

So let's implement this. Okay, so first up, let me scroll to where we built the dataset and let's change the block size from 3 to 8. So we're going to be taking eight characters of context to predict the ninth character. So the dataset now looks like this — we have a lot more context feeding in to predict any next character in a sequence, and these eight characters are going to be processed in this tree-like structure.

### Re-running Baseline Code on block_size 8

Now, if we scroll here, everything should just be able to work. So we should be able to redefine the network. You see the number of parameters has increased by 10,000, and that's because the block size has grown, so this first linear layer is much, much bigger. Our linear layer now takes eight characters into this middle layer, so there's a lot more parameters there. But this should just run — let me just break right after the very first iteration. So you see that this runs just fine; it's just that this network doesn't make too much sense — we're crushing way too much information way too fast.

So let's now come in and see how we could try to implement the hierarchical scheme. Now, before we dive into the detail of the re-implementation here, I was just curious to actually run it and see where we are in terms of the baseline performance of just lazily scaling up the context length. So I'll let it run — we get a nice loss curve — and then evaluating the loss, we actually see quite a bit of improvement just from increasing the context length.

So I started a little bit of a performance log here, and previously where we were is we were getting a performance of 2.10 on the validation loss. And now, simply scaling up the context length from 3 to 8 gives us a performance of 2.02 — so quite a bit of an improvement here. And also, when you sample from the model, you see that the names are definitely improving qualitatively as well.

So we could of course spend a lot of time here tuning things and making it even bigger and scaling up the network further even with the simple sort of setup here, but let's continue and let's implement the WaveNet model and treat this as just a rough baseline performance. But there's a lot of optimization left on the table in terms of some of the hyperparameters that you're hopefully getting a sense of now.

### Implementing the WaveNet Architecture

Okay, so let's scroll up now and come back up. What I've done here is I've created a bit of a scratch space for us to just look at the forward pass of the neural net and inspect the shape of the tensor along the way as the neural net forwards. So here I'm just temporarily, for debugging, creating a batch of just say four examples — so four random integers, then I'm plucking out those rows from our training set, and then I'm passing into the model the input `xb`.

Now the shape of `xb` here, because we have only four examples, is 4 by 8, and this 8 is now the current block size. So inspecting `xb`, we just see that we have four examples, each one of them is a row of `xb`, and we have eight characters here. And this integer tensor just contains the identities of those characters.

So the first layer of our neural net is the embedding layer. So passing `xb` — this integer tensor — through the embedding layer creates an output that is 4 by 8 by 10. So our embedding table has, for each character, a 10-dimensional vector that we are trying to learn. And so what the embedding layer does here is it plucks out the embedding vector for each one of these integers and organizes it all in a 4 by 8 by 10 tensor. So all of these integers are translated into 10-dimensional vectors inside this three-dimensional tensor.

Now, passing that through the flatten layer — as you recall, what this does is it views this tensor as just a 4 by 80 tensor. And what that effectively does is that all these 10-dimensional embeddings for all these eight characters just end up being stretched out into a long row. And that looks kind of like a concatenation operation, basically. So by viewing the tensor differently, we now have a 4 by 80, and inside this 80 it's all the 10-dimensional vectors just concatenated next to each other.

And then the linear layer of course takes 80 and creates 200 channels just via matrix multiplication. So far so good.

Now I'd like to show you something surprising. Let's look at the insides of the linear layer and remind ourselves how it works. The linear layer here in the forward pass takes the input `x`, multiplies it with a weight, and then optionally adds bias. And the weight here is two-dimensional as defined here, and the bias is one-dimensional.

So effectively, in terms of the shapes involved, what's happening inside this linear layer looks like this right now — and I'm using random numbers here but I'm just illustrating the shapes and what happens. Basically, a 4 by 80 input comes into the linear layer, that's multiplied by this 80 by 200 weight matrix inside, and there's a plus 200 bias, and the shape of the whole thing that comes out of the linear layer is 4 by 200 as we see here.

Now notice, by the way, that this here will create a 4×200 tensor, and then plus 200 — there's a broadcasting happening here. A 4 by 200 broadcasts with 200, so everything works here.

So now the surprising thing that I'd like to show you that you may not expect is that this input here that is being multiplied doesn't actually have to be two-dimensional. This matrix multiply operator in PyTorch is quite powerful, and in fact you can actually pass in higher-dimensional arrays or tensors and everything works fine. So for example, this could be 4 by 5 by 80, and the result in that case will become 4 by 5 by 200. You can add as many dimensions as you like on the left here.

And so effectively what's happening is that the matrix multiplication only works on the last dimension, and the dimensions before it in the input tensor are left unchanged. So basically, these dimensions on the left are all treated as just a batch dimension. So we can have multiple batch dimensions, and then in parallel over all those dimensions we are doing the matrix multiplication on the last dimension.

So this is quite convenient because we can use that in our network now. Because remember that we have these eight characters coming in, and we don't want to now flatten all of it out into a large eight-dimensional vector. Because we don't want to matrix multiply 80 into a weight matrix immediately. Instead, we want to group these — like this — so every consecutive two elements: one and two, three and four, five and six, seven and eight. All of these should be now basically flattened out and multiplied by a weight matrix, but all of these four groups here we'd like to process in parallel. So it's kind of like a batch dimension that we can introduce.

And then we can in parallel basically process all of these bigram groups in the four batch dimensions of an individual example, and also over the actual batch dimension of the, you know, four examples in our example here.

So let's see how that works. Effectively, what we want is: right now we take a 4 by 80 and multiply it by 80 by 200 in the linear layer — this is what happens. But instead, what we want is we don't want 80 characters or 80 numbers to come in, we only want two characters to come in on the very first layer, and those two characters should be fused. So in other words, we just want 20 to come in — 20 numbers would come in. And here we don't want a 4 by 80 to feed into the linear layer, we actually want these groups of two to feed in. So instead of 4 by 80, we want this to be a 4 by 4 by 20. So these are the four groups of two, and each one of them is a 10-dimensional vector.

So what we want now is we need to change the flatten layer so it doesn't output a 4 by 80 but it outputs a 4 by 4 by 20, where basically every two consecutive characters are packed in on the very last dimension. And then this 4 is the first batch dimension and this 4 is the second batch dimension, referring to the four groups inside every one of these examples. And then this will just multiply like this. So this is what we want to get to.

So we're going to have to change the linear layer in terms of how many inputs it expects — it shouldn't expect 80, it should just expect 20 numbers. And we have to change our flatten layer so it doesn't just fully flatten out this entire example; it needs to create a 4 by 4 by 20 instead of 4 by 80.

So let's see how this could be implemented. Basically, right now we have an input that is a 4 by 8 by 10 that feeds into the flatten layer, and currently the flatten layer just stretches it out. So if you remember the implementation of `Flatten`, it takes `x` and it just views it as whatever the batch dimension is and then negative one.

So effectively what it does right now is `e.view(4, -1)` and the shape of this of course is 4 by 80. So that's what currently happens, and we instead want this to be a 4 by 4 by 20 where these consecutive 10-dimensional vectors get concatenated.

So, you know how in Python you can take a `list(range(10))`? So we have numbers from zero to nine, and we can index like this to get all the even parts. And we can also index starting at one and going in steps of two to get all the odd parts.

So one way to implement this would be as follows: we can take `e` and we can index into it for all the batch elements, and then just even elements in this dimension — so at indexes 0, 2, 4, and 6 — and then all the parts from this last dimension. And this gives us the even characters. And then this gives us all the odd characters.

And basically what we want to do is we want to make sure that these get concatenated in PyTorch. And then we want to concatenate these two tensors along the second dimension. So this — and the shape of it would be 4 by 4 by 20. This is definitely the result we want: we are explicitly grabbing the even parts and the odd parts and we're arranging those 4 by 4 by 10 right next to each other and concatenating.

So this works, but it turns out that what also works is you can simply use a `view` again and just request the right shape. And it just so happens that in this case those vectors will again end up being arranged in exactly the way we want. So in particular, if we take `e` and we just view it as a 4 by 4 by 20 — which is what we want — we can check that this is exactly equal. So let me call this "explicit" — the explicit concatenation, I suppose.

So `explicit.shape` is 4 by 4 by 20. If you just view it as 4 by 4 by 20, you can check that when you compare to `explicit` — this is an element-wise operation — making sure that all of them are true: that is true.

So basically, long story short, we don't need to make an explicit call to concatenate, etc. We can simply take this input tensor to flatten and we can just view it in whatever way we want. And in particular, you don't want to stretch things out with negative one; we want to actually create a three-dimensional array. And depending on how many vectors that are consecutive we want to fuse — like, for example, two — then we can just simply ask for this dimension to be 20. And use a negative 1 here and PyTorch will figure out how many groups it needs to pack into this additional batch dimension.

So let's now go into `Flatten` and implement this. Okay, so I scroll up here to `Flatten` and what we'd like to do is we'd like to change it now. So let me create a constructor and take the number of elements that are consecutive that we would like to concatenate in the last dimension of the output.

So here we're just going to remember `self.n = n`. And then I want to be careful here because PyTorch actually has a `torch.nn.Flatten` and its keyword arguments are different and they kind of function differently. So our `Flatten` is going to start to depart from PyTorch `Flatten`, so let me call it `FlattenConsecutive` or something like that, just to make sure that our APIs are roughly equal.

So this basically flattens only some `n` consecutive elements and puts them into the last dimension.

Now, here the shape of `x` is B by T by C. So let me pop those out into variables, and recall that in our example down below, B was 4, T was 8, and C was 10.

Now, instead of doing `x.view(B, -1)` — right, this is what we had before — we want this to be `B` by negative 1 by, and basically here we want `C * n` — that's how many consecutive elements we want.

And here, instead of negative one, I don't super love the use of negative one because I like to be very explicit so that you get error messages when things don't go according to your expectation. So what do we expect here? We expect this to become `T // n`, using integer division here. So that's what I expect to happen.

And then one more thing I want to do here is: remember, previously, all the way in the beginning, `n` was three, and basically we're concatenating all the three characters that existed there. So we basically are concatenating everything.

And so sometimes I can create a spurious dimension of one here. So if it is the case that `x.shape[1]` is 1, then it's kind of like a spurious dimension. We don't want to return a three-dimensional tensor with a 1 here; we just want to return a two-dimensional tensor exactly as we did before.

So in this case, basically we will just say `x = x.squeeze()` — that is a PyTorch function. And `squeeze` takes a dimension that it either squeezes out — all the dimensions of a tensor that are one — or you can specify the exact dimension that you want to be squeezed. And again, I like to be as explicit as possible always, so I expect to squeeze out the first dimension only of this tensor, this three-dimensional tensor. And if this dimension here is 1, then I just want to return B by `C * n`.

And so `self.out = x`, and then we return `self.out`. So that's the candidate implementation. And of course this should be `self.n` instead of just `n`.

So let's run. And let's come here now and take it for a spin. So `FlattenConsecutive`, and in the beginning let's just use 8 — so this should recover the previous behavior. So `FlattenConsecutive` of 8, which is the current block size — we can do this — that should recover the previous behavior.

So we should be able to run the model. And here we can inspect — I have a little code snippet here where I iterate over all the layers, I print the name of this class and the shape.

And so we see the shapes as we expect them after every single layer in the network. So now let's try to restructure it using our `FlattenConsecutive` and do it hierarchically.

So in particular, we want to `FlattenConsecutive` not block_size but just 2. And then we want to process this with `Linear` — now then, the number of inputs to this linear will not be `n_embd * block_size`; it will now only be `n_embd * 2` — 20. This goes through the first layer.

And now we can in principle just copy-paste this. Now the next linear layer should expect `n_hidden * 2`, and the last piece of it should expect `n_hidden * 2` again.

So this is sort of like the naive version of it. Running this, we now have a much, much bigger model. And we should be able to basically just forward the model.

And now we can inspect the numbers in between. So 4 by 8 by 10 was flattened consecutively into 4 by 4 by 20. This was projected into 4 by 4 by 200. And then BatchNorm just worked out of the box — we have to verify that BatchNorm does the correct thing even though it takes a three-dimensional input instead of a two-dimensional input. Then we have tanh which is element-wise.

Then we crushed it again — so we flattened consecutively and ended up with a 4 by 2 by 400 now. Then linear brought it back down to 200, BatchNorm, tanh. And lastly we get a 4 by 400, and we see that the `FlattenConsecutive` for the last flatten here squeezed out that dimension of one, so we only ended up with 4 by 400. And then linear, BatchNorm, tanh, and the last linear layer to get our logits. And so the logits end up in the same shape as they were before, but now we actually have a nice three-layer neural net.

And it basically corresponds to — whoops, sorry — it basically corresponds exactly to this network now, except only this piece here because we only have three layers, whereas here in this example there's four layers with the total receptive field size of 16 characters instead of just 8 characters. So the block size here is 16. So this piece of it is basically implemented here.

Now we just have to kind of figure out some good channel numbers to use here. In particular, I changed the number of hidden units to be 68 in this architecture, because when I use 68 the number of parameters comes out to be 22,000 — so that's exactly the same that we had before. And we have the same amount of capacity in this neural net in terms of the number of parameters, but the question is whether we are utilizing those parameters in a more efficient architecture.

### Training the WaveNet: First Pass

So what I did then is I got rid of a lot of the debugging cells here and I reran the optimization. And scrolling down to the result, we see that we get the identical performance roughly. So our validation loss now is 2.029, and previously it was 2.027. So controlling for the number of parameters, changing from the flat to hierarchical is not giving us anything yet.

That said, there are two things to point out. Number one, we didn't really torture the architecture here very much — this is just my first guess, and there's a bunch of hyperparameter search that we could do in terms of how we allocate our budget of parameters to what layers. Number two, we still may have a bug inside the BatchNorm1d layer. So let's take a look at that, because it runs but does it do the right thing?

### Fixing the BatchNorm1d Bug

So I pulled up the layer inspector that we have here and printed out the shape along the way. And currently it looks like the BatchNorm is receiving an input that is 32 by 4 by 68. And here on the right I have the current implementation of BatchNorm that we have right now.

Now, this BatchNorm assumed — in the way we wrote it, and at the time — that `x` is two-dimensional. So it was N by D, where N was the batch size, so that's why we only reduced the mean and the variance over the zeroth dimension. But now `x` will basically become three-dimensional.

So what's happening inside the BatchNorm right now, and how come it's working at all and not giving any errors? The reason for that is basically because everything broadcasts properly, but the BatchNorm is not doing what we want it to do.

So in particular, let's basically think through what's happening inside the BatchNorm. I have the code here. So we're receiving an input of 32 by 4 by 68, and then we are doing `x.mean` — here I have `e` instead of `x`, but we're doing the mean over 0, and that's actually giving us 1 by 4 by 68. So we're doing the mean only over the very first dimension, and it's giving us a mean and a variance that still maintain this dimension here.

So these means are only taking over 32 numbers in the first dimension. And then when we perform this, everything broadcasts correctly still. But basically, what ends up happening is — when we also look at the running mean, the shape of it — so I'm looking at `model.layers[3]`, which is the first BatchNorm layer, and looking at whatever the running mean became and its shape — the shape of this running mean now is 1 by 4 by 68.

Right, instead of it being just a size of dimension — because we have 68 channels, we expect to have 68 means and variances that we're maintaining — but actually we have an array of 4 by 68. And so basically what this is telling us is: this BatchNorm is currently working in parallel over 4 times 68 instead of just 68 channels.

So basically, we are maintaining statistics for every one of these four positions individually and independently. And instead, what we want to do is we want to treat this 4 as a batch dimension, just like the zeroth dimension. So as far as the BatchNorm is concerned, we don't want to average over 32 numbers — we want to now average over 32 times 4 numbers for every single one of these 68 channels.

And so let me now fix this. It turns out that when you look at the documentation of `torch.mean` — so let's go to `torch.mean` — in one of its signatures, when we specify the dimension, we see that the dimension here is not just an int; it can also be a tuple of ints. So we can reduce over multiple integers at the same time — over multiple dimensions at the same time. So instead of just reducing over 0, we can pass in a tuple `(0, 1)`. And here `(0, 1)` as well.

And then what's going to happen is the output of course is going to be the same, but now what's going to happen is because we reduce over 0 and 1, if we look at `xmean.shape`, we see that now we've reduced — we took the mean over both the zeroth and the first dimension. So we're just getting 68 numbers and a bunch of spurious dimensions here.

So now this becomes 1 by 1 by 68, and the running mean and the running variance analogously will become 1 by 1 by 68. So even though there are spurious dimensions, the correct thing will happen in that we are only maintaining means and variances for 68 channels. And we're calculating the mean and variance across 32 times 4 dimensions, so that's exactly what we want.

And let's change the implementation of `BatchNorm1d` that we have so that it can take in two-dimensional or three-dimensional inputs and perform accordingly. So at the end of the day, the fix is relatively straightforward. Basically, the dimension we want to reduce over is either 0 or the tuple (0, 1), depending on the dimensionality of `x`. So if `x.ndim` is 2 — so it's a two-dimensional tensor — then the dimension we want to reduce over is just the integer 0. Elif `x.ndim` is 3 — so it's a three-dimensional tensor — then the dims we're going to assume are (0, 1) that we want to reduce over. And then here we just pass in `dim`.

And if the dimensionality of `x` is anything else, we'll now get an error, which is good.

So that should be the fix. Now I want to point out one more thing — we're actually departing from the API of PyTorch here a little bit. Because when you come to `BatchNorm1d` in PyTorch, you can scroll down and you can see that the input to this layer can either be N by C — where N is the batch size and C is the number of features or channels — or it actually does accept three-dimensional inputs, but it expects it to be N by C by L, where L is like the sequence length or something like that.

So this is a problem, because you see how C is nested here in the middle. And so when it gets three-dimensional inputs, this BatchNorm layer will reduce over 0 and 2 instead of 0 and 1. So basically, PyTorch's `BatchNorm1d` layer assumes that C will always be the first dimension, whereas we assume here that C is the last dimension and there are some number of batch dimensions beforehand. It expects N by C or N by C by L; we expect N by C or N by L by C. And so it's a deviation. I think it's okay — I prefer it this way honestly. So this is the way that we will keep it for our purposes.

### Re-training WaveNet with Bug Fix

So I redefined the layers, re-initialized the neural net, and did a single forward pass with a break just for one step. Looking at the shapes along the way, they're of course identical — all the shapes are the same. But the way we see that things are actually working as we want them to now is that when we look at the BatchNorm layer, the running mean shape is now 1 by 1 by 68. So we're only maintaining 68 means for every one of our channels, and we're treating both the zeroth and the first dimension as a batch dimension, which is exactly what we want.

So let me retrain the neural net now. Okay, so I retrained the neural net with the bug fix — we get a nice curve, and when we look at the validation performance, we do actually see a slight improvement. So we went from 2.029 to 2.022. So basically, the bug inside the BatchNorm was holding us back a little bit, it looks like, and we are getting a tiny improvement now. But it's not clear if this is statistically significant.

And the reason we slightly expect an improvement is because we're not maintaining so many different means and variances that are only estimated using 32 numbers. Effectively, now we are estimating them using 32 times 4 numbers, so you just have a lot more numbers that go into any one estimate of the mean and variance, and it allows things to be a bit more stable and less wiggly inside those estimates of those statistics.

### Scaling Up Our WaveNet

So pretty nice. With this more general architecture in place, we are now set up to push the performance further by increasing the size of the network. So for example, I bumped up the number of embeddings to 24 instead of 10, and also increased the number of hidden units. But using the exact same architecture, we now have 76,000 parameters, and the training takes a lot longer. But we do get a nice curve, and then when you actually evaluate the performance, we are now getting validation performance of 1.993.

So we've crossed over the 2.0 sort of territory and right about 1.99, but we are starting to have to wait quite a bit longer. And we're a little bit in the dark with respect to the correct setting of the hyperparameters here and the learning rates and so on, because the experiments are starting to take longer to train. And so we are missing sort of like an experimental harness on which we could run a number of experiments and really tune this architecture very well.

## Conclusions

### Experimental Harness

So I'd like to conclude now with a few notes. We basically improved our performance from a starting of 2.1 down to 1.99, but I don't want that to be the focus, because honestly we're kind of in the dark. We have no experimental harness, we're just guessing and checking, and this whole thing is terrible. We're just looking at the training loss — normally you want to look at both the training and the validation loss together, and the whole thing looks different if you're actually trying to squeeze out numbers.

That said, we did implement this architecture from the WaveNet paper, but we did not implement the specific forward pass of it where you have a more complicated linear layer sort of — this gated linear layer kind — and there's residual connections and skip connections and so on. So we did not implement that; we just implemented this structure.

### WaveNet but with Dilated Causal Convolutions

I would like to briefly hint or preview how what we've done here relates to convolutional neural networks as used in the WaveNet paper. And basically, the use of convolutions is strictly for efficiency — it doesn't actually change the model we've implemented.

So here, for example, let me look at a specific name to work with an example. So there's a name in our training set and it's "DeAndre," and it has seven letters. So that is eight independent examples in our model — all these rows here are independent examples of "DeAndre."

Now, you can forward of course any one of these rows independently. So I can take my model and call it on any individual index. Notice, by the way, here I'm being a little bit tricky. The reason for this is that `x_train[7]` — that shape is just a one-dimensional array of 8. So you can't actually call the model on it, you're going to get an error because there's no batch dimension. So when you do `x_train[[7]]`, then the shape of this becomes 1 by 8, so I get an extra batch dimension of 1, and then we can forward the model.

So that forwards a single example. And you might imagine that you actually may want to forward all of these eight at the same time. So pre-allocating some memory and then doing a for loop eight times and forwarding all of those eight here will give us all the logits in all these different cases.

Now, for us with the model as we've implemented it right now, this is eight independent calls to our model. But what convolutions allow you to do is they allow you to basically slide this model efficiently over the input sequence. And so this for loop can be done not outside in Python but inside of kernels in CUDA, and so this for loop gets hidden into the convolution.

So the convolution basically — you can think of it as a for loop applying a little linear filter over space of some input sequence. And in our case, the space we're interested in is one-dimensional, and we're interested in sliding these filters over the input data.

So this diagram is actually fairly good as well. Basically, what we've done is — here they are highlighting in black one individual sort of tree of this calculation, so just calculating the single output example here. And so this is basically what we've implemented here — we've implemented a single, this black structure, and calculated a single output, like a single example.

But what convolutions allow you to do is they allow you to take this black structure and kind of slide it over the input sequence here and calculate all of these orange outputs at the same time. Or here, that corresponds to calculating all of these outputs at all the positions of "DeAndre" at the same time.

And the reason that this is much more efficient is because number one, as I mentioned, the for loop is inside the CUDA kernels in the sliding, so that makes it efficient. But number two, notice the variable reuse here. For example, if we look at this node — this node here is the right child of this node but is also the left child of the node here. And so basically, this node and its value is used twice. And so right now in this naive way we'd have to recalculate it, but here we are allowed to reuse it.

So in the convolutional neural network, you think of these linear layers that we have up above as filters, and we take these filters — they're linear filters — and you slide them over the input sequence. And we calculate the first layer and then the second layer and then the third layer and then the output layer of the sandwich, and it's all done very efficiently using these convolutions.

So we're going to cover that in a future video.

### torch.nn

The second thing I hope you took away from this video is you've seen me basically implement all of these layer Lego building blocks or module building blocks, and I'm implementing them over here. We've implemented a number of layers together and we've also implemented these containers, and we've overall PyTorchified our code quite a bit more.

Now, basically what we're doing here is we're re-implementing `torch.nn`, which is the neural networks library on top of `torch.tensor`. And it looks very much like this, except it is much better because it's in PyTorch instead of jangling in my Jupyter notebook. So I think going forward, I will probably have considered us having unlocked `torch.nn`. We understand roughly what's in there, how these modules work, how they're nested, and what they're doing on top of `torch.tensor`. So hopefully we'll just switch over and continue and start using `torch.nn` directly.

### The Development Process of Building Deep Neural Nets

The next thing I hope you got a bit of a sense of is what the development process of building deep neural networks looks like, which I think was relatively representative to some extent.

So number one, we are spending a lot of time in the documentation page of PyTorch, and we're reading through all the layers, looking at documentation, what are the shapes of the inputs, what can they be, what does the layer do, and so on. Unfortunately, I have to say the PyTorch documentation is not very good. They spend a ton of time on hardcore engineering of all kinds of distributed primitives, etc., but as far as I can tell, no one is maintaining any documentation. It will lie to you, it will be wrong, it will be incomplete, it will be unclear. So unfortunately it is what it is, and you just kind of do your best with what they've given us.

Number two, the other thing that I hope you got a sense of is there's a ton of trying to make the shapes work, and there's a lot of gymnastics around these multi-dimensional arrays — are they two-dimensional, three-dimensional, four-dimensional? What layers take what shapes? Is it NCL or NLC? And you're permuting and viewing, and it just can get pretty messy.

And so that brings me to number three: I very often prototype these layers and implementations in Jupyter notebooks and make sure that all the shapes work out. And I'm spending a lot of time basically babysitting the shapes and making sure everything is correct. And then once I'm satisfied with the functionality in the Jupyter notebook, I will take that code and copy-paste it into my repository of actual code that I'm training with. And so then I'm working with VS Code on the side. So I usually have Jupyter notebook and VS Code — I develop in Jupyter notebook, I paste into VS Code, and then I kick off experiments from the code repository.

So that's roughly some notes on the development process of working with neural nets.

### Going Forward

Lastly, I think this lecture unlocks a lot of potential further lectures. Because number one, we have to convert our neural network to actually use these dilated causal convolutional layers — so implementing the ConvNet. Number two, potentially starting to get into what residual connections and skip connections mean and why they are useful.

Number three, as I mentioned, we don't have any experimental harness. So right now I'm just guessing and checking everything — this is not representative of typical deep learning workflows. You have to set up your evaluation harness, you can kick off experiments, you have lots of arguments that your script can take, you're kicking off a lot of experimentation, you're looking at a lot of plots of training and validation losses, and you're looking at what is working and what is not working. And you're working on this population level and you're doing all these hyperparameter searches. And so we've done none of that so far, so how to set that up and how to make it good I think is a whole other topic.

Number three, we should probably cover recurrent neural networks — RNNs, LSTMs, GRUs — and of course Transformers. So many places to go, and we'll cover that in the future.

### Improve on My Loss!

For now, bye — sorry, I forgot to say that if you are interested, I think it is kind of interesting to try to beat this number 1.993, because I really haven't tried a lot of experimentation here and there's quite a bit of fruit potentially to still pick further.

So I haven't tried any other ways of allocating these channels in this neural net — maybe the number of dimensions for the embedding is all wrong. Maybe it's possible to actually take the original network with just one hidden layer and make it big enough and actually beat my fancy hierarchical network. It's not obvious — that would be kind of embarrassing if this did not do better even once you torture it a little bit.

Maybe you can read the WaveNet paper and try to figure out how some of these layers work and implement them yourselves using what we have. And of course you can always tune some of the initialization or some of the optimization and see if you can improve it that way.

So I'd be curious if people can come up with some ways to beat this. And yeah, that's it for now. Bye!
