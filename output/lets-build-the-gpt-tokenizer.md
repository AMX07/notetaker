# Let's Build the GPT Tokenizer

*From Andrej Karpathy's "Neural Networks: Zero to Hero" series*

---

## Introduction: Tokenization, the GPT-2 Paper, and Tokenization-Related Issues

Hi everyone so in this video I'd like us to cover the process of tokenization in large language models now you see here that I have a sad face and that's because tokenization is my least favorite part of working with large language models but unfortunately it is necessary to understand in some detail because it it is fairly hairy gnarly and there's a lot of hidden footguns to be aware of and a lot of oddness with large language models typically traces back to tokenization

So what is tokenization now in my previous video Let's Build GPT from scratch we actually already did tokenization but we did a very naive simple version of tokenization so when you go to the Google colab for that video you see here that we loaded our training set and our training set was this Shakespeare dataset now in the beginning the Shakespeare dataset is just a large string in Python it's just text and so the question is how do

We plug text into large language models and in this case here we created a vocabulary of 65 possible characters that we saw occur in this string these were the possible characters and we saw that there are 65 of them and then we created a a lookup table for converting from every possible character a little string piece into a token an integer so here for example we tokenized the string High there and we received this sequence of tokens

And here we took the first 1,000 characters of our dataset and we encoded it into tokens and because it is this is character level we received 1,000 tokens in a sequence so token 18 47 etc. now later we saw that the way we plug these tokens into the language model is by using an embedding table and so basically if we have 65 possible tokens then this embedding table is going to have 65 rows and roughly speaking we're taking

The integer associated with every single single token we're using that as a lookup into this table and we're plucking out the corresponding row and this row is a is trainable parameters that we're going to train using back propagation and this is the vector that then feeds into the Transformer and that's how the Transformer sort of perceives every single token so here we had a very naive tokenization process that was a character level tokenizer but in practice

In state-of-the-art language models people use a lot more complicated schemes unfortunately for constructing these token vocabularies so we're not dealing on the Character level we're dealing on chunk level and the way these character chunks are constructed is using algorithms such as for example the Byte Pair Encoding algorithm which we're going to go into in detail and cover in this video I'd like to briefly show you the paper that introduced a byte-level encoding as a mechanism for tokenization

In the context of large language models and I would say that that's probably the GPT-2 paper and if you scroll down here to the section input representation this is where they cover tokenization the kinds of properties that you'd like the tokenization to have and they conclude here that they're going to have a tokenizer where you have a vocabulary of 50,257 possible tokens and the context size is going to be 1,024 tokens so in the in

In the attention layer of the Transformer neural network every single token is attending to the previous tokens in the sequence and it's going to see up to 1,024 tokens so tokens are this like fundamental unit the atom of large language models if you will and everything is in units of tokens everything is about tokens and tokenization is the process for translating strings or text into sequences of tokens and vice versa when you go into the Llama 2 paper as well I can show you that when you search token you're going to get get 63 hits

And that's because tokens are again pervasive so here they mentioned that they trained on two trillion tokens of data and so on so we're going to build our own tokenizer luckily the Byte Pair Encoding algorithm is not that super complicated and we can build it from scratch ourselves and we'll see exactly how this works before we dive into code I'd like to give you a brief Taste of some of the complexities that come from the tokenization because I just want to make sure that

We motivate it sufficiently for why we are doing all this and why this is so gross so tokenization is at the heart of a lot of weirdness in large language models and I would advise that you do not brush it off a lot of the issues that may look like just issues with the new network architecture or the large language model itself are actually issues with the tokenization and fundamentally Trace back to it so if you've noticed any issues with large language models can't you know not able to do spelling tasks very easily that's usually due to tokenization simple string processing can be difficult

For the large language model to perform natively non-english languages can work much worse and to a large extent this is due to tokenization sometimes LLMs are bad at simple arithmetic also can trace be traced to tokenization GPT-2 specifically would have had quite a bit more issues with python than future versions of it due to tokenization there's a lot of other issues maybe you've seen weird warnings about a trailing whitespace this is a tokenization issue if you had asked GPT earlier about SolidGoldMagikarp

And what it is you would see the LLM go totally crazy and it would start going off about a completely unrelated tangent topic maybe you've been told to use YAML over JSON in structure data all of that has to do with tokenization so basically tokenization is at the heart of many issues I will look back around to these

## Tokenization by Example in a Web UI (tiktokenizer)

At the end of the video but for now let me just skip over it a little bit and let's go to this web app the tiktokenizer vercel.app so I have it loaded here and what I like about this web app is that tokenization is running a sort of live in your browser in JavaScript so you can just type here stuff hello world and the whole string tokenizes so here what we see on the left is a string that you put

In on the right we're currently using the GPT-2 tokenizer we see that this string that I pasted here is currently tokenizing into 300 tokens and here they are sort of shown explicitly in different colors for every single token so for example this word tokenization became two tokens the token 3,642 and 1,634 the token space is is token 318 so be careful on the bottom you can show whitespace and keep in mind that there are spaces and \n new line characters

In here but you can hide them for clarity the token space at is token 379 the to the Token space the is 262 etc. so you notice here that the space is part of that token chunk now so this is kind of like how our English sentence broke up and that seems all well and good now now here I put in some arithmetic so we see that the token 127 Plus and then token six space 6 followed by 77

So what's happening here is that 127 is feeding in as a single token into the large language model but the number 677 will actually feed in as two separate tokens and so the large language model has to sort of take account of that and process it correctly in its Network and see here 804 will be broken up into two tokens and it's is all completely arbitrary and here I have another example of four-digit numbers and they break up

In a way that they break up and it's totally arbitrary sometimes you have multiple digits single token sometimes you have individual digits as many tokens and it's all kind of pretty arbitrary and coming out of the tokenizer here's another example we have the string egg and you see here that this became two tokens but for some reason when I say I have an egg you see when it's a space egg it's two token it's sorry it's a single token

So just egg by itself in the beginning of a sentence is two tokens but here as a space egg is suddenly a single token for the exact same string okay here lowercase egg turns out to be a single token and in particular notice that the color is different so this is a different token so this is case sensitive and of course a capital egg would also be different tokens and again this would be two tokens arbitrarily

So so for the same concept egg depending on if it's in the beginning of a sentence at the end of a sentence lowercase uppercase or mixed all this will be basically very different tokens and different IDs and the language model has to learn from raw data from all the internet text that it's going to be training on that these are actually all the exact same concept and it has to sort of group them in the parameters of

The neural network and understand just based on the data patterns that these are all very similar but maybe not almost exactly similar but but very very similar after the EG demonstration here I have an introduction from OpenAI's chbt in Korean so annyeonghaseyo etc. so this is in Korean and the reason I put this here is because you'll notice that non-english languages work slightly worse in ChatGPT part of this is because of course the training dataset

For ChatGPT is much larger for English and for everything else but the same is true not just for the large language model itself but also for the tokenizer so when we train the tokenizer we're going to see that there's a training set as well and there's a lot more English than non-english and what ends up happening is that we're going to have a lot more longer tokens for English so how do I put this if you have a single sentence

In English and you tokenize it you might see that it's 10 tokens or something like that but if you translate that sentence into say Korean or Japanese or something else you'll typically see that the number of tokens used is much larger and that's because the chunks here are a lot more broken up so we're using a lot more tokens for the exact same thing and what this does is it bloats up the sequence length of all

The documents so you're using up more tokens and then in the attention of the Transformer when these tokens try to attend each other you are running out of context in the maximum context length of that Transformer and so basically all the non-english text is stretched out from the perspective of the Transformer and this just has to do with the trainings that used for the tokenizer and the tokenization itself so it will create a lot bigger tokens

And a lot larger groups in English and it will have a lot of little boundaries for all the other non-english text so if we translated this into English it would be significantly fewer tokens the final example I have here is a little snippet of python for doing FizzBuzz and what I'd like you to notice is look all these individual spaces are all separate tokens they are token 220 so 220 220 220 220 and then space

If is a single token and so what's going on here is that when the Transformer is going to consume or try to create this text it needs to handle all these spaces individually they all feed in one by one into the entire Transformer in the sequence and so this is being extremely wasteful tokenizing it in this way and so as a result of that GPT-2 is not very good with python and it's not anything to do with coding or

The language model itself it's just that if he use a lot of indentation using space in Python like we usually do you just end up bloating out all the text and it's separated across way too much of the sequence and we are running out of the context length in the sequence that's roughly speaking what's what's happening we're being way too wasteful we're taking up way too much token space now we can also scroll up here and

We can change the tokenizer so note here that GPT-2 tokenizer creates a token count of 300 for this string here we can change it to cl100k_base which is the GPT for tokenizer and we see that the token count drops to 185 so for the exact same string we are now roughly having the number of tokens and roughly speaking this is because the number of tokens in the GPT 4 tokenizer is roughly double that of

The number of tokens in the GPT-2 tokenizer so we went went from roughly 50k to roughly 100K now you can imagine that this is a good thing because the same text is now squished into half as many tokens so this is a lot denser input to the Transformer and in the Transformer every single token has a finite number of tokens before it that it's going to pay attention to and so what this is doing is we're roughly able to see twice as much text as a context

For what token to predict next because of this change but of course just increasing the number of tokens is not strictly better infinitely because as you increase the number of tokens now your embedding table is sort of getting a lot larger and also at the output we are trying to predict the next token and there's the softmax there and that grows as well we're going to go into more detail later on this but there's some kind of a Sweet Spot somewhere where you have a just right number of tokens

In your vocabulary where everything is appropriately dense and still fairly efficient now one thing I would like you to note specifically for the GPT-4 tokenizer is that the handling of the whitespace for python has improved a lot you see that here these four spaces are represented as one single token for the three spaces here and then the token SPF and here seven spaces were all grouped into a single token so we're being a lot more efficient

In how we represent Python and this was a deliberate Choice made by OpenAI when they designed the GPT-4 tokenizer and they group a lot more space into a single character what this does is this densifies Python and therefore we can attend to more code before it when we're trying to predict the next token in the sequence and so the Improvement in the python coding ability from GPT-2 to GPT-4 is not just a matter of the language model

And the architecture and the details of the optimization but a lot of the Improvement here is also coming from the design of the tokenizer and how it groups characters into tokens okay so

## Strings in Python and Unicode Code Points

Let's now start writing some code so remember what we want to do we want to take strings and feed them into language models for that we need to somehow tokenize strings into some integers in some fixed vocabulary and then we will use those integers to make a look up into a lookup table of vectors and feed those vectors into the Transformer as an input now the reason this gets a little bit tricky of course is that

We don't just want to support the simple English alphabet we want to support different kinds of languages so this is annyeonghaseyo in Korean which is hello and we also want to support many kinds of special characters that we might find on the internet for example Emoji so how do we feed this text into Transformers well how's the what is this text anyway in Python so if you go to the documentation of a string in Python you can see that strings are immutable sequences of Unicode code points

Okay what are Unicode code points we can go to PDF so Unicode code points are defined by the Unicode Consortium as part of the Unicode standard and what this is really is that it's just a definition of roughly 150,000 characters right now and roughly speaking what they look like and what integers represent those characters so it says 150,000 characters across 161 scripts as of right now so if you scroll down here you can see that the standard is very much alive

The latest standard 15.1 in September 2023 and basically this is just a way to define lots of types of characters like for example all these characters across different scripts so the way we can access the Unicode code point given Single Character is by using the ord() function in Python so for example I can call ord('H') and I can see that for the Single Character H the Unicode code point is 104 okay but this can be arbitr complicated

So we can take for example our Emoji here and we can see that the code point for this one is 128,000 or we can take un and this is 50,000 now keep in mind you can't plug in strings here because you this doesn't have a single code point it only takes a single Unicode code point character and tells you its integer so in this way we can look up all the characters of this specific string and their code points

So ord(x) for x in this string and we get this encoding here now see here we've already turned the raw code points already have integers so why can't we simply just use these integers and not have any tokenization at all why can't we just use this natively as is and just use the code point well one reason for that of course is that the vocabulary in that case would be quite long so in this case

For Unicode the this is a vocabulary of 150,000 different code points but more worryingly than that I think the Unicode standard is very much alive and it keeps changing and so it's not kind of a stable representation necessarily that we may want to use directly so for those reasons we need something a bit better

## Unicode Byte Encodings: ASCII, UTF-8, UTF-16, UTF-32

So to find something better we turn to encodings so if we go to the Wikipedia page here we see that the Unicode consortion defines three types of encodings UTF-8 UTF-16 and UTF-32 these encoding are the way by which we can take Unicode text and translate it into binary data or byte streams UTF-8 is by far the most common so this is the UTF-8 page now this Wikipedia page is actually quite long but what's important for our purposes is that UTF-8 takes every single code point

And it translates it to a byte stream and this byte stream is between one to four bytes so it's a variable length encoding so depending on the Unicode point according to the schema you're going to end up with between 1 to four bytes for each code point on top of that there's UTF-8 UTF-16 and UTF-32 UTF-32 is nice because it is fixed length instead of variable length but it has many other downsides as well so the full kind of spectrum of pros

And cons of all these different three encodings are beyond the scope of this video I just like to point out that I enjoyed this blog post and this blog post at the end of it also has a number of references that can be quite useful one of them is UTF-8 everywhere Manifesto and this Manifesto describes the reason why UTF-8 is significantly preferred and a lot nicer than the other encodings and why it is used a lot more prominently on

The internet one of the major advantages just just to give you a sense is that UTF-8 is the only one of these that is backwards compatible to the much simpler ASCII encoding of text but I'm not going to go into the full detail in this video so suffice to say that we like the UTF-8 encoding and let's try to take the string and see what we get if we encoded into UTF-8 the string class in Python actually has do encode

And you can give it the encoding which is say UTF-8 now we get out of this is not very nice because this is the bytes is a bytes object and it's not very nice in the way that it's printed so I personally like to take it through list because then we actually get the raw B of this encoding so this is the raw bytes that represent this string according to the UTF-8 encoding we can also look at UTF-16

We get a slightly different byte stream and we here we start to see one of the disadvantages of UTF-16 you see how we have zero Z something Z something Z something we're starting to get a sense that this is a bit of a wasteful encoding and indeed for simple ASCII characters or English characters here we just have the structure of 0 something Z something and it's not exactly nice same for UTF-32 when we expand this we can start to get a sense of

The wastefulness of this encoding for our purposes you see a lot of zeros followed by something and so this is not desirable so suffice it to say that we would like to stick with UTF-8 for our purposes however if we just use UTF-8 naively these are byte streams so that would imply a vocabulary length of only 256 possible tokens but this this vocabulary size is very very small what this is going to do if we just were to use it naively is that all of our text would be stretched out over very very long sequences of bytes

And so what what this does is that certainly the embedding table is going to be tiny and the prediction at the top at the final layer is going to be very tiny but our sequences are very long and remember that we have pretty finite context length and the attention that we can support in a transformer for computational reasons and so we only have as much context length but now we have very very long sequences and this is just inefficient

And it's not going to allow us to attend to sufficiently long text before us for the purposes of the next token prediction task so we don't want to use the raw bytes of the UTF-8 encoding we want to be able to support larger vocabulary size that we can tune as a hyper but we want to stick with the UTF-8 encoding of these strings so what do we do well the answer of course is we turn to

The Byte Pair Encoding algorithm which will allow us to compress these byte sequences to a variable amount so we'll get to that in a bit but I just want to briefly speak to the fact that I

## Daydreaming: Deleting Tokenization

Would love nothing more than to be able to feed raw byte sequences into language models in fact there's a paper about how this could potentially be done from Summer last last year now the problem is you actually have to go in and you have to modify the Transformer architecture because as I mentioned you're going to have a problem where the attention will start to become extremely expensive because the sequences are so long and so in this paper they propose kind of a hierarchical structuring of

The Transformer that could allow you to just feed in raw bytes and so at the end they say together these results establish the viability of tokenization free autoregressive sequence modeling at scale so tokenization free would indeed be amazing we would just feed B streams directly into our models but unfortunately I don't know that this has really been proven out yet by sufficiently many groups and a sufficient scale but something like this at one point would be amazing

And I hope someone comes up with it but for now we have to come back and we can't feed this directly into language models and we have to compress it using the Byte Pair Encoding algorithm so let's see how that works so as I mentioned the B

## Byte Pair Encoding (BPE) Algorithm Walkthrough

Paare encoding algorithm is not all that complicated and the Wikipedia page is actually quite instructive as far as the basic idea goes go what we're doing is we have some kind of a input sequence like for example here we have only four elements in our vocabulary a b c and d and we have a sequence of them so instead of bytes let's say we just have four a vocab size of four the sequence is too long

And we'd like to compress it so what we do is that we iteratively find the pair of tokens that occur the most frequently and then once we've identified that pair we repl replace that pair with just a single new token that we append to our vocabulary so for example here the byte pair AA occurs most often so we mint a new token let's call it capital Z and we replace every single occurrence of AA by Z

So now we have two Z's here so here we took a sequence of 11 characters with vocabulary size four and we've converted it to a sequence of only nine tokens but now with a vocabulary of five because we have a fifth vocabulary element that we just created and it's Z standing for concatination of AA and we can again repeat this process so we again look at the sequence and identify the pair of tokens that are most frequent let's say that that is

Now AB well we are going to replace AB with a new token that we mint call Y so y becomes ab and then every single occurrence of ab is now replaced with y so we end up with this so now we only have 1 2 3 4 5 6 seven characters in our sequence but we have not just four vocabulary elements or five but now we have six and for the final round we again look through

The sequence find that the phrase zy or the pair zy is most common and replace it one more time with another character let's say x so X is z y and we replace all curses of zy and we get this following sequence so basically after we have gone through this process instead of having a sequence of 11 tokens with a vocabulary length of four we now have a sequence of 1 2 3 four five tokens but our vocabulary length

Now is seven and so in this way we can iteratively compress our sequence I we mint new tokens so in the in the exact same way we start we start out with byte sequences so we have 256 vocabulary size but we're now going to go through these and find the byte pairs that occur the most and we're going to iteratively start minting new tokens appending them to our vocabulary and replacing things and in this way we're going to end up with a compressed training dataset

And also an algorithm for taking any arbitrary sequence and encoding it using this vocabul and also decoding it back to Strings so let's now Implemint all that so here's

## Starting the Implementation

What I did I went to this blog post that I enjoyed and I took the first paragraph and I copy pasted it here into text so this is one very long line here now to get the tokens as I mentioned we just take our text and we encode it into UTF-8 the tokens here at this point will be a raw bytes single stream of bytes and just so that it's easier to work with instead of just a bytes object I'm going to convert all those bytes to integers

And then create a list of it just so it's easier for us to manipulate and work with in Python and visualize and here I'm printing all of that so this is the original this is the original paragraph and its length is 533 code points and then here are the bytes encoded in ut UTF-8 and we see that this has a length of 616 bytes at this point or 616 tokens and the reason this is more is because a lot of these simple ASCII characters or simple characters they just become a single byte

But a lot of these Unicode more complex characters become multiple bytes up to four and so we are expanding that size so now what we'd like to do as a first step of the algorithm is we'd like to iterate over here and find the pair of bytes that occur most frequently because we're then going to merge it so if you are working long on a notebook on a side then I encourage you to basically click on

The link find this notebook and try to write that function yourself otherwise I'm going to come here and Implement first the function that finds the most common pair okay so here's what

## Counting Consecutive Pairs and Finding the Most Common Pair

I came up with there are many different ways to implement this but I'm calling the function get stats it expects a list of integers I'm using a dictionary to keep track of basically the counts and then this is a pythonic way to iterate consecutive elements of this list which we covered in the previous video and then here I'm just keeping track of just incrementing by one for all the pairs so if I call this on all

The tokens here then the stats comes out here so this is the dictionary the keys are these tuples of consecutive elements and this is the count so just to print it in a slightly better way this is one way that I like to do that where you it's a little bit compound here so you can pause if you like but we iterate all all the items the items called on dictionary returns pairs of key value and instead I create a list

Here of value key because if it's a value key list then I can call sort on it and by default python will use the first element which in this case will be value to sort by if it's given tuples and then reverse so it's descending and print that so basically it looks like 101 comma 32 was the most commonly occurring consecutive pair and it occurred 20 times we can double check that that makes reasonable sense so

If I just search 101, 32 then you see that these are the 20 occurrences of that pair and if we'd like to take a look at what exactly that pair is we can use chr which is the opposite of ord() in Python so we give it a Unicode code point so 101 and of 32 and we see that this is e and space so basically there's a lot of E space here meaning that a lot of these words seem to end with e

So here's eace as an example so there's a lot of that going on here and this is the most common pair

## Merging the Most Common Pair

So now that we've identified the most common pair we would like to iterate over this sequence we're going to mint a new token with the ID of 256 right because these tokens currently go from Z to 255 so when we create a new token it will have an ID of 256 and we're going to iterate over this entire list and every every time we see 101 comma 32 we're going to swap that out for 256 so let's Implement that

Now and feel free to do that yourself as well so first I commented this just so we don't pollute the notebook too much this is a nice way of in Python obtaining the highest ranking pair so we're basically calling the Max on this dictionary stats and this will return the maximum key and then the question is how does it rank keys so you can provide it with a function that ranks keys and that function is just stats.

Getet stats.get would basically return the value and so we're ranking by the value and getting the maximum key so it's 101 comma 32 as we saw now to actually merge 101, 32 this is the function that I wrote but again there are many different versions of it so we're going to take a list of IDs and the the pair that we want to replace and that pair will be replaced with the new index idx so iterating through IDs

If we find the pair swap it out for idx so we create this new list and then we start at zero and then we go through this entire list sequentially from left to right and here we are checking for equality at the current position with the pair so here we are checking that the pair matches now here is a bit of a tricky condition that you have to append if you're trying to be careful and that is that you don't want

This here to be out of Bounds at the very last position when you're on the rightmost element of this list otherwise this would give you an autof bounds error so we have to make sure that we're not at the very very last element so this would be false for that so if we find a match we append to this new list that replacement index and we increment the position by two so we skip over that entire pair

But otherwise if we we haven't found a matching pair we just sort of copy over the elemint at that position and increment by one then return this so here's a very small toy example if we have a list 566 791 and we want to replace the occurrences of 67 with 99 then calling this on that will give us what we're asking for so here the 67 is replaced with 99 so now I'm going to uncomment this

For our actual use case where we want to take our tokens we want to take the top pair here and replace it with 256 to get tokens to if we run this we get the following so recall that previously we had a length 616 in this list and now we have a length 596 right so this decreased by 20 which makes sense because there are 20 occurrences moreover we can try to find 256 here and we see plenty of occurrences on off it

And moreover just double check there should be no occurrence of 101, 32 so this is the original array plenty of them and in the second array there are no occurrences of 101, 32 so we've successfully merged this single pair and now we just iterate this so we are going to go over the sequence again find the most common pair and replace it so let me now write a while loop that uses these functions to do this sort of iteratively

And how many times do we do it four well that's totally up to us as a hyperparameter the more steps we take the larger will be our vocabulary and the shorter will be our sequence and there is some sweet spot that we usually find works the best in practice and so this is kind of a hyperparameter and we tune it and we find good vocabulary sizes as an example GPT-4 currently uses roughly 100,000 tokens and ballpark that those are reasonable numbers currently instead

The are large language models so let me now write putting putting it all together and iterating these steps

## Training the Tokenizer: Adding the While Loop and Compression Ratio

Okay now before we dive into the while loop I wanted to add one more cell here where I went to the blog post and instead of grabbing just the first paragraph or two I took the entire blog post and I stretched it out in a single line and basically just using longer text will allow us to have more representative statistics for the byte Pairs and we'll just get a more sensible results out of it because it's longer text

So here we have the raw text we encode it into bytes using the UTF-8 encoding and then here as before we are just changing it into a list of integers in Python just so it's easier to work with instead of the raw bytes objects and then this is the code that I came up with to actually do the merging in Loop these two functions here are identical to what we had above I only included them here just

So that you have the point of reference here so these two are identical and then this is the new code that I added so the first first thing we want to do is we want to decide on the final vocabulary size that we want our tokenizer to have and as I mentioned this is a hyperparameter and you set it in some way depending on your best performance so let's say for us we're going to use 276 because that way we're going to be doing exactly 20 merges

And 20 merges because we already have 256 tokens for the raw bytes and to reach 276 we have to do 20 merges to add 20 new tokens here this is one way in Python to just create a copy of a list so I'm taking the tokens list and by wrapping it in a list python will construct a new list of all the individual elements so this is just a copy operation then here I'm creating a merges dictionary

So this merges dictionary is going to maintain basically the child one child two mapping to a new token and so what we're going to be building up here is a binary tree of merges but actually it's not exactly a tree because a tree would have a single root node with a bunch of leaves for us we're starting with the leaves on the bottom which are the individual bytes those are the starting 256 tokens and then we're starting to like merge two of them at a time

And so it's not a tree it's more like a forest as we merge these elements so for 20 merges we're going to find the most commonly occurring pair we're going to mint a new token integer for it so I will start at zero so we'll going to start at 256 we're going to print that we're merging it and we're going to replace all of the occurrences of that pair with the new newly minted token and we're going to record that

This pair of integers merged into this new integer so running this gives us the following output so we did 20 merges and for example the first merge was exactly as before the 101, 32 tokens merging into a new token 256 now keep in mind that the individual tokens 101 and 32 can still occur in the sequence after merging it's only when they occur exactly consecutively that that becomes 256 now and in particular the other thing to notice

Here is that the token 256 which is the newly minted token is also eligible for merging so here on the bottom the 20th merge was a merge of 25 and 259 becoming 275 so every time we replace these tokens they become eligible for merging in the next round of iteration so that's why we're building up a small sort of binary Forest instead of a single individual tree one thing we can take a look at as well is

We can take a look at the compression ratio that we've achieved so in particular we started off with this tokens list so we started off with 24,000 bytes and after merging 20 times we now have only 19,000 tokens and so therefore the compression ratio simply just dividing the two is roughly 1.27 so that's the amount of compression we were able to achieve of this text with only 20 merges and of course the more vocabulary elements you add

The greater the compression ratio here would be finally so that's kind of like the

## The Tokenizer/LLM Diagram: It Is a Completely Separate Stage

Training of the tokenizer if you will now 1 Point I wanted to make is that and maybe this is a diagram that can help kind of illustrate is that tokenizer is a completely separate object from the large language model itself so everything in this lecture we're not really touching the LLM itself we're just training the tokenizer this is a completely separate pre-processing stage usually so the tokenizer will have its own training set just like a large language model has a potentially different training set

So the tokenizer has a training set of documents on which you're going to train the tokenizer and then and we're performing The Byte pair encoding algorithm as we saw above to train the vocabulary of this tokenizer so it has its own training set it is a pre-processing stage that you would run a single time in the beginning and the tokenizer is trained using Byte Pair Encoding algorithm once you have the tokenizer once it's trained and you have

The vocabulary and you have the merges we can do both encoding and decoding so these two arrows here so the tokenizer is a translation layer between raw text which is as we saw the sequence of Unicode code points it can take raw text and turn it into a token sequence and vice versa it can take a token sequence and translate it back into raw text so now that we have trained tokenizer and we have these merges

We are going to turn to how we can do the encoding and the decoding step if you give me text here are the tokens and vice versa if you give me tokens here's the text once we have that we can translate between these two Realms and then the language model is going to be trained as a step two afterwards and typically in a in a sort of a state-of-the-art application you might take all of your training data

For the language model and you might run it through the tokenizer and sort of translate everything into a massive token sequence and then you can throw away the raw text you're just left with the tokens themselves and those are stored on disk and that is what the large language model is actually reading when it's training on them so this one approach that you can take as a single massive pre-processing step a stage so yeah basically I think

The most important thing I want to get across is that this is completely separate stage it usually has its own entire training set you may want to have those training sets be different between the tokenizer and the large language model so for example when you're training the tokenizer as I mentioned we don't just care about the performance of English text we care about multi many different languages and we also care about code or not code so you may want to look into different kinds of mixtures of different kinds of languages

And different amounts of code and things like that because the amount of different language that you have in your tokenizer training set will determine how many merges of it there will be and therefore that determines the density with which this type of data is sort of has in the token space and so roughly speaking intuitively if you add some amount of data like say you have a ton of Japanese data in your tokenizer training set then that means that more Japanese tokens will get merged

And therefore Japanese will have shorter sequences and that's going to be beneficial for the large language model which has a finite context length on which it can work on in in the token space so hopefully that makes sense so we're now going to turn to encoding and decoding now that we have trained a tokenizer so we have our merges and now how do we do encoding and decoding okay

## Decoding Tokens to Strings

So let's begin with decoding which is this Arrow over here so given a token sequence let's go through the tokenizer to get back a python string object so the raw text so this is the function that we' like to implement we're given the list of integers and we want to return a python string if you'd like try to implement this function yourself it's a fun exercise otherwise I'm going to start pasting in my own solution so there are many different ways to do it here's

One way I will create an kind of pre-processing variable that I will call vocab and vocab is a mapping or a dictionary in Python for from the token ID to the bytes object for that token so we begin with the raw bytes for tokens from 0 to 255 and then we go in order of all the merges and we sort of populate this vocab list by doing an addition here so this is the basically the bytes representation of

The first child followed by the second one and remember these are bytes objects so this addition here is an addition of two bytes objects just concatenation so that's what we get here one tricky thing to be careful with by the way is that I'm iterating a dictionary in Python using a DOT items and it really matters that this runs in the order in which we inserted items into the merges dictionary luckily starting with python 3.7 this is guaranteed to be

The case but before python 3.7 this iteration may have been out of order with respect to how we inserted elements into merges and this may not have worked but we are using an modern python so we're okay and then here given the IDS the first thing we're going to do is get the tokens so the way I implemented this here is I'm taking I'm iterating over all the IDS I'm using vocab to look up their bytes

And then here this is one way in Python to concatenate all these bytes together to create our tokens and then these tokens here at this point are raw bytes so I have to decode using UTF F now back into python strings so previously we called that encode on a string object to get the bytes and now we're doing it Opposite we're taking the bytes and calling a decode on the bytes object to get a string in Python

And then we can return text so this is how we can do it now this actually has a issue in the way I implemented it and this could actually throw an error so try to think figure out why this code could actually result in an error if we plug in some sequence of IDs that is unlucky so let me demonstrate the issue when I try to decode just something like 97 I am going to get letter A

Here back so nothing too crazy happening but when I try to decode 128 as a single element the token 128 is what in string or in Python object Unicode decoder UTF-8 can't Decode by 0x8 which is this in HEX in position zero invalid start byte what does that mean well to understand what this means we have to go back to our UTF-8 page that I briefly showed earlier and this is Wikipedia UTF-8 and basically there's a specific schema that UTF-8 bytes take

So in particular if you have a multi-te object for some of the Unicode characters they have to have this special sort of envelope in how the encoding works and so what's happening here is that invalid start byte that's because 128 the binary representation of it is one followed by all zeros so we have one and then all zero and we see here that that doesn't conform to the format because one followed by all zero just doesn't fit any of these rules

So to speak so it's an invalid start byte which is byte one this one must have a one following it and then a zero following it and then the content of your Unicode in hex here so basically we don't exactly follow the UTF-8 standard and this cannot be decoded and so the way to fix this is to use this errors equals in bytes. decode function of python and by default errors is strict so we will throw an error

If it's not valid UTF-8 bytes encoding but there are many different things that you could put here on error handling this is the full list of all the errors that you can use and in particular instead of strict let's change it to replace and that will replace with this special marker this replacement character so errors equals replace and now we just get that character back so basically not every single by sequence is valid UTF-8 and if it happens that your large language model

For example predicts your tokens in a bad manner then they might not fall into valid UTF-8 and then we won't be able to decode them so the standard practice is to basically use errors equals replace and this is what you will also find in the OpenAI code that they released as well but basically whenever you see this kind of a character in your output in that case something went wrong and the LM output not was not valid sort of sequence of

## Encoding Strings to Tokens

Tokens okay and now we're going to go the other way so we are going to implement this Arrow right here where we are going to be given a string and we want to encode it into tokens so this is the signature of the function that we're interested in and this should basically print a list of integers of the tokens so again try to maybe implement this yourself if you'd like a fun exercise and pause here otherwise I'm going to start putting

In my solution so again there are many ways to do this so this is one of the ways that sort of I came came up with so the first thing we're going to do is we are going to take our text encode it into UTF-8 to get the raw bytes and then as before we're going to call list on the bytes object to get a list of integers of those bytes so those are the starting tokens those are

The raw bytes of our sequence but now of course according to the merges dictionary above and recall this was the merges some of the bytes may be merged according to this lookup in addition to that remember that the merges was built from top to bottom and this is sort of the order in which we inserted stuff into merges and so we prefer to do all these merges in the beginning before we do these merges later because

For example this merge over here relies on the 256 which got merged here so we have to go in the order from top to bottom sort of if we are going to be merging anything now we expect to be doing a few merges so we're going to be doing W true and now we want to find a pair of bytes that is consecutive that we are allowed to merge according to this in order to reuse some of

The functionality that we've already written I'm going to reuse the function get stats so recall that get stats will give us the we'll basically count up how many times every single pair occurs in our sequence of tokens and return that as a dictionary and the dictionary was a mapping from all the different by pairs to the number of times that they occur right at this point we don't actually care how many times they occur in the sequence

We only care what the raw pairs are in that sequence and so I'm only going to be using basically the keys of the dictionary I only care about the set of possible merge candidates if that makes sense now we want to identify the pair that we're going to be merging at this stage of the loop so what do we want we want to find the pair or like the a key inside stats that has the lowest index

In the merges dictionary because we want to do all the early merges before we work our way to the late merges so again there are many different ways to implement this but I'm going to do something a little bit fancy here so I'm going to be using the Min over an iterator in Python when you call Min on an iterator and stats here as a dictionary we're going to be iterating the keys of this dictionary in Python

So we're looking at all the pairs inside stats which are all the consecutive Pairs and we're going to be taking the consecutive pair inside tokens that has the minimum what the Min takes a key which gives us the function that is going to return a value over which we're going to do the Min and the one we care about is we're we care about taking merges and basically getting that pairs index so basically for any pair inside stats

We are going to be looking into merges at what index it has and we want to get the pair with the Min number so as an example if there's a pair 101 and 32 we definitely want to get that pair we want to identify it here and return it and pair would become 101, 32 if it occurs and the reason that I'm putting a float INF here as a fall back is that in the get function when

We call when we basically consider a pair that doesn't occur in the merges then that pair is not eligible to be merged right so if in the token sequence there's some pair that is not a merging pair it cannot be merged then it doesn't actually occur here and it doesn't have an index and it cannot be merged which we will denote as float INF and the reason Infinity is nice here is because for sure we're guaranteed that it's not going to participate

In the list of candidates when we do the men so so this is one way to do it so B basically long story short this Returns the most eligible merging candidate pair that occurs in the tokens now one thing to be careful with here is this function here might fail in the following way if there's nothing to merge then then there's nothing in merges that satisfi that is satisfied anymore there's nothing to merge everything just returns float imps

And then the pair I think will just become the very first element of stats but this pair is not actually a mergeable pair it just becomes the first pair inside stats arbitrarily because all of these pairs evaluate to float in for the merging Criterion so basically it could be that this this doesn't look succeed because there's no more merging pairs so if this pair is not in merges that was returned then this is a signal for us that actually there was nothing to merge no single pair can be merged anymore

In that case we will break out nothing else can be merged you may come up with a different implementation by the way this is kind of like really trying hard in Python but really we're just trying to find a pair that can be merged with the lowest index here now if we did find a pair that is inside merges with the lowest index then we can merge it so we're going to look into the merger dictionary

For that pair to look up the index and we're going to now merge that into that index so we're going to do tokens equals and we're going to replace the original tokens we're going to be replacing the pair pair and we're going to be replacing it with index idx and this returns a new list of tokens where every occurrence of pair is replaced with idx so we're doing a merge and we're going to be continuing this until eventually nothing can be merged we'll come out

Here and we'll break out and here we just return tokens and so that that's the implementation I think so hopefully this runs okay cool yeah and this looks reasonable so for example 32 is a space in ASCII so that's here so this looks like it worked great okay so let's wrap up this section of the video at least I wanted to point out that this is not quite the right implementation just yet because we are leaving out a special case

So in particular if we try to do this this would give us an error and the issue is that if we only have a single character or an empty string then stats is empty and that causes an issue inside Min so one way to fight this is if L of tokens is at least two because if it's less than two it's just a single token or no tokens then let's just there's nothing to merge so we just return

So that would fix that case Okay and then second I have a few test cases here for us as well so first let's make sure about or let's note the following if we take a string and we try to encode it and then decode it back you'd expect to get the same string back right is that true for all strings so I think so here it is the case and I think in general this is probably

The case but notice that going backwards is not is not you're not going to have an identity going backwards because as I mentioned us not all token sequences are valid UTF-8 sort of byte streams and so so therefore you're some of them can't even be decodable so this only goes in One Direction but for that one direction we can check here if we take the training text which is the text that we train to tokenizer around

We can make sure that when we encode and decode we get the same thing back which is true and here I took some validation data so I went to I think this web page and I grabbed some text so this is text that the tokenizer has not seen and we can make sure that this also works okay so that gives us some confidence that this was correctly implemented so those are the basics of the Byte Pair Encoding algorithm

We saw how we can take some training set train a tokenizer the parameters of this tokenizer really are just this dictionary of merges and that basically creates the little binary Forest on top of raw bytes once we have this the merges table we can both encode and decode between raw text and token sequences so that's the the simplest setting of The tokenizer what we're going to do now though is we're going to look at some of

The St the art lar language models and the kinds of tokenizers that they use and we're going to see that this picture complexifies very quickly so we're going to go through the details of this comp complexification one at a time so let's

## Regex Patterns to Force Splits Across Categories

Kick things off by looking at the GPT Series so in particular I have the GPT-2 paper here and this paper is from 2019 or so so 5 years ago and let's scroll down to input representation this is where they talk about the tokenizer that they're using for GPT-2 now this is all fairly readable so I encourage you to pause and read this yourself but this is where they motivate the use of the Byte Pair Encoding algorithm on

The byte level representation of UTF-8 encoding so this is where they motivate it and they talk about the vocabulary sizes and everything now everything here is exactly as we've covered it so far but things start to depart around here so what they mention is that they don't just apply the naive algorithm as we have done it and in particular here's a example suppose that you have common words like dog what will happen is that dog of course occurs very frequently

In the text and it occurs right next to all kinds of punctuation as an example so doc dot dog exclamation mark dog question mark etc. and naively you might imagine that the BPE algorithm could merge these to be single tokens and then you end up with lots of tokens that are just like dog with a slightly different punctuation and so it feels like you're clustering things that shouldn't be clustered you're combining kind of semantics with uation

And this feels suboptimal and indeed they also say that this is suboptimal according to some of the experiments so what they want to do is they want to top down in a manual way enforce that some types of characters should never be merged together so they want to enforce these merging rules on top of the Byte Pair Encoding algorithm so let's take a look at their code and see how they actually enforce this and what kinds of mergy they actually do perform

So I have to to tab open here for GPT-2 under OpenAI on GitHub and when we go to Source there is an encoder.py now I don't personally love that they call it encoder.py because this is the tokenizer and the tokenizer can do both encode and decode so it feels kind of awkward to me that it's called encoder but that is the tokenizer and there's a lot going on here and we're going to step through it

In detail at one point for now I just want to focus on this part here the create a regex pattern here that looks very complicated and we're going to go through it in a bit but this is the core part that allows them to enforce rules for what parts of the text Will Never Be merged for sure now notice that re. compile here is a little bit misleading because we're not just doing import re which is

The python re module we're doing import reex as re and reex is a python package that you can install pip install r x and it's basically an extension of re so it's a bit more powerful re so let's take a look at this pattern and what it's doing and why this is actually doing the separation that they are looking for okay so I've copy pasted the pattern here to our jupit notebook where we left off and let's take

This pattern for a spin so in the exact same way that their code does we're going to call an re. findall for this pattern on any arbitrary string that we are interested so this is the string that we want to encode into tokens to feed into n LLM like GPT-2 so what exactly is this doing well re. findall will take this pattern and try to match it against a string the way this works is that you are going from left to right

In the string and you're trying to match the pattern and re.findall will get all the occurrences and organize them into a list now when you look at the when you look at this pattern first of all notice that this is a raw string and then these are three double quotes just to start the string so really the string itself this is the pattern itself right and notice that it's made up of a lot of ores so see these vertical bars those are ores

In regex and so you go from left to right in this pattern and try to match it against the string wherever you are so we have hello and we're going to try to match it well it's not apostrophe s it's not apostrophe t or any of these but it is an optional space followed by- P of sorry SLp{L} one or more times what is/p{L} it is coming to some documentation that I found there might be other sources as well \p{L} is a letter any kind of letter from any language

And hello is made up of letters h e l etc. so optional space followed by a bunch of letters one or more letters is going to match hello but then the match ends because a whitespace is not a letter so from there on begins a new sort of attempt to match against the string again and starting in here we're going to skip over all of these again until we get to the exact same Point again and

We see that there's an optional space this is the optional space followed by a bunch of letters one or more of them and so that matches so when we run this we get a list of two elements hello and then space world so how are you if we add more letters we would just get them like this now what is this doing and why is this important we are taking our string and instead of directly encoding it

For tokenization we are first splitting it up and when you actually step through the code and we'll do that in a bit more detail what really is doing on a high level is that it first splits your text into a list of texts just like this one and all these elements of this list are processed independently by the tokenizer and all of the results of that processing are simply concatenated so hello world oh I I missed how hello world how are you

We have five elements of list all of these will independent independently go from text to a token sequence and then that token sequence is going to be concatenated it's all going to be joined up and roughly speaking what that does is you're only ever finding merges between the elements of this list so you can only ever consider merges within every one of these elements in individually and after you've done all the possible merging for all of these elements individually

The results of all that will be joined by concatenation and so you are basically what what you're doing effectively is you are never going to be merging this e with this space because they are now parts of the separate elements of this list and so you are saying we are never going to merge eace because we're breaking it up in this way so basically using this regx pattern to Chunk Up the text is just one way of enforcing that some merges are not to happen

And we're going to go into more of this text and we'll see that what this is trying to do on a high level is we're trying to not merge across letters across numbers across punctuation and so on so let's see in more detail how that works so let's continue now we have/ p{N} if you go to the documentation \p{L} of n is any kind of numeric character in any script so it's numbers so we have an optional space followed by numbers

And those would be separated out so letters and numbers are being separated so if I do Hello World 123 how are you then world will stop matching here because one is not a letter anymore but one is a number so this group will match for that and we'll get it as a separate entity let's see how these apostrophes work so here if we have Slash V or I mean apostrophe V as an example then apostrophe here is not a letter or a number

So hello will stop matching and then we will exactly match this with that so that will come out as a separate thing so why are they doing the apostrophes here honestly I think that these are just like very common apostrophes p that are used typically I don't love that they've done this because let me show you what happens when you have some Unicode apostrophes like for example you can have if you have house then this will be separated out because of

This matching but if you use the Unicode apostrophe like this then suddenly this does not work and so this apostrophe will actually become its own thing now and so so it's basically hardcoded for this specific kind of apostrophe and otherwise they become completely separate tokens in addition to this you can go to the GPT-2 docs and here when they Define the pattern they say should have added re. ignore case so BPE merges can happen for capitalized versions of contractions

So what they're pointing out is that you see how this is apostrophe and then lowercase letters well because they didn't do re. ignore case then then these rules will not separate out the apostrophes if it's uppercase so house would be like this but if I did house if I'm uppercase then notice suddenly the apostrophe comes by itself so the tokenization will work differently in uppercase and lower case inconsistently separating out these apostrophes so it feels extremely gnarly

And slightly gross but that's that's how that works okay so let's come back after trying to match a bunch of apostrophe Expressions by the way the other issue here is that these are quite language specific probably so I don't know that all the languages for example use or don't use apostrophes but that would be inconsistently tokenized as a result then we try to match letters then we try to match numbers and then if that doesn't work

We fall back to here and what this is saying is again optional space followed by something that is not a letter number or a space in one or more of that so what this is doing effectively is this is trying to match punctuation roughly speaking not letters and not numbers so this group will try to trigger for that so if I do something like this then these parts here are not letters or numbers but they will actually they are they will actually get caught

Here and so they become its own group so we've separated out the punctuation and finally this this is also a little bit confusing so this is matching whitespace but this is using a negative look ahead assertion in regex so what this is doing is it's matching whitespace up to but not including the last whitespace character why is this important this is pretty subtle I think so you see how the whitespace is always included at the beginning of

The word so space r space u etc. suppose we have a lot of spaces here what's going to happen here is that these spaces up to not including the last character will get caught by this and what that will do is it will separate out the spaces up to but not including the last character so that the last character can come here and join with the space you and the reason that's nice is because space you is

The common token so if I didn't have these Extra Spaces here you would just have space you and if I add tokens if I add spaces we still have a space view but now we have all this extra whitespace so basically the GB to tokenizer really likes to have a space letters or numbers and it it preens these spaces and this is just something that it is consistent about so that's what that is for and then finally

We have all the the last fallback is whitespace characters so that would be just if that doesn't get caught then this thing will catch any trailing spaces and so on I wanted to show one more real world example here so if we have this string which is a piece of python code and then we try to split it up then this is the kind of output we get so you'll notice that the list has many elements

Here and that's because we are splitting up fairly often every time sort of a category changes so there will never be any merges Within These elements and that's what you are seeing here now you might think that in order to train the tokenizer OpenAI has used this to split up text into chunks and then run just a BPE algorithm within all the chunks but that is not exactly what happened and the reason is the following notice that

We have the spaces here those Spaces end up being entire elements but these spaces never actually end up being merged by by OpenAI and the way you can tell is that if you copy paste the exact same chunk here into tiktokenizer you see that all the spaces are kept independent and they're all token 220 so I think OpenAI at some point Point enforce some rule that these spaces would never be merged and so there's some additional rules on top of just chunking

And BPE that OpenAI is not clear about now the training code for the GPT-2 tokenizer was never released so all we have is the code that I've already shown you but this code here that they've released is only the inference code for the tokens so this is not the training code you can't give it a piece of text and training tokenizer this is just the inference code which Tak takes the merges that we have up above

And applies them to a new piece of text and so we don't know exactly how OpenAI trained train the tokenizer but it wasn't as simple as chunk it up and BPE it whatever it was

## The tiktoken Library: Differences Between GPT-2 and GPT-4 Regex

Next I wanted to introduce you to the tiktoken library from OpenAI, which is the official library for tokenization from OpenAI so this is tiktoken pip install pip install tiktoken and then you can do the tokenization in inference this is again not training code this is only inference code for tokenization I wanted to show you how you would use it quite simple and running this just gives us the GPT-2 tokens or the GPT 4 tokens so this is

The tokenizer use for GPT 4 and so in particular we see that the whitespace in GPT-2 remains unmerged but in GPT 4 these whitespaces merge as we also saw in this one where here they're all unmerged but if we go down to GPT 4 they become merged now in the GPT-4 tokenizer they changed the regular expression that they use to Chunk Up text so the way to see this is that if you come to your the tiktoken library

And then you go to this file tiktoken X OpenAI public this is where sort of like the definition of all these different tokenizers that OpenAI maintains is and so necessarily to do the inference they had to publish some of the details about the strings so this is the string that we already saw for GPT-2 it is slightly different but it is actually equivalent to what we discussed here so this pattern that we discussed is equivalent to

This pattern this one just executes a little bit faster so here you see a little bit of a slightly different definition but otherwise it's the same we're going to go into special tokens in a bit and then if you scroll down to cl100k_base this is the GPT 4 tokenizer you see that the pattern has changed and this is kind of like the main the major change in addition to a bunch of other special tokens which I'll go into

In a bit again now some I'm not going to actually go into the full detail of the pattern change because honestly this is my numbing I would just advise that you pull out ChatGPT and the regex documentation and just step through it but really the major changes are number one you see this eye here that means that the case sensitivity this is case insensitive match and so the comment that we saw earlier on oh we should have used re.

Uppercase basically we're now going to be matching these apostrophe s apostrophe D apostrophe M etc. we're going to be matching them both in lowercase and in uppercase so that's fixed there's a bunch of different like handling of the whitespace that I'm not going to go into the full details of and then one more thing here is you will notice that when they match the numbers they only match one to three numbers so so they will never merge numbers that are

In low in more than three digits only up to three digits of numbers will ever be merged and that's one change that they made as well to prevent tokens that are very very long number sequences but again we don't really know why they do any of this stuff because none of this is documented and it's just we just get the pattern so yeah it is what it is but those are some of the changes that GPT-4 has made

And of course the vocabulary size went from roughly 50k to roughly 100K the next thing I would like to do

## GPT-2 encoder.py Released by OpenAI: Walkthrough

Very briefly is to take you through the GPT-2 encoder.py that OpenAI has released this is the file that I already mentioned to you briefly now this file is fairly short and should be relatively understandable to you at this point starting at the bottom here they are loading two files encoder.json and vocab.bpe and they do some light processing on it and then they call this encoder object which is the tokenizer now if you'd like to inspect these two files which together constitute their saved tokenizer then you can do that with a piece of code like

This this is where you can download these two files and you can inspect them if you'd like and what you will find is that this encoder as they call it in their code is exactly equivalent to our vocab so remember here where we have this vocab object which allowed us us to decode very efficiently and basically it took us from the integer to the bytes for that integer so our vocab is exactly their encoder and then their vocab.bpe confusingly is actually are merges

So their BPE merges which is based on the data inside vocab.bpe ends up being equivalent to our merges so basically they are saving and loading the two variables that for us are also critical the merges variable and the vocab variable using just these two variables you can represent a tokenizer and you can both do encoding and decoding once you've trained this tokenizer now the only thing that is actually slightly confusing inside what OpenAI does here is that

In addition to this encoder and a decoder they also have something called a byte encoder and a byte decoder and this is actually unfortunately just kind of a spurious implementation detail and isn't actually deep or interesting in any way so I'm going to skip the discussion of it but what OpenAI does here for reasons that I don't fully understand is that not only have they this tokenizer which can encode and decode but they have a whole separate layer

Here in addition that is used serially with the tokenizer and so you first do byte encode and then encode and then you do decode and then byte decode so that's the loop and they are just stacked serial on top of each other and and it's not that interesting so I won't cover it and you can step through it if you'd like otherwise this file if you ignore the byte encoder and the byte decoder will be algorithmically very familiar with you

And the meat of it here is the what they call BPE function and you should recognize this Loop here which is very similar to our own while loop where they're trying to identify the bigram a pair that they should be merging next and then here just like we had they have a for Loop trying to merge this pair so they will go over all of the sequence and they will merge the pair whenever they find it

And they keep repeating that until they run out of possible merges in the in the text so that's the meat of this file and there's an encode and a decode function just like we have implemented it so long story short what I want you to take away at this point is that unfortunately it's a little bit of a messy code that they have but algorithmically it is identical to what we've built up above and what we've built up above

If you understand it is algorithmically what is necessary to actually build a BPE tokenizer train it and then both encode and decode

## Special Tokens and tiktoken Handling: GPT-2/GPT-4 Differences

The next topic I would like to turn to is that of special tokens so in addition to tokens that are coming from you know raw bytes and the BPE merges we can insert all kinds of tokens that we are going to use to delimit different parts of the data or introduced to create a special structure of the token streams so in if you look at this encoder object from OpenAI's GPT-2 right here we mentioned this is very similar to our vocab you'll

Notice that the length of this is 50257 and as I mentioned it's mapping and it's inverted from the mapping of our vocab our vocab goes from integer to string and they go the other way around for no amazing reason but the thing to note here is that this the mapping table here is 50257 where does that number come from where what are the tokens as I mentioned there are 256 raw byte token tokens and then OpenAI actually did 50,000 merges

So those become the other tokens but this would have been 50256 so what is the 57th token and there is basically one special token and that one special token you can see is called <|endoftext|> so this is a special token and it's the very last token and this token is used to delimit documents ments in the training set so when we're creating the training data we have all these documents and we tokenize them and

We get a stream of tokens those tokens only range from Z to 50256 and then in between those documents we put special <|endoftext|> token and we insert that token in between documents and we are using this as a signal to the language model that the document has ended and what follows is going to be unrelated to the document previously that said the language model has to learn this from data it it needs to learn that

This token usually means that it should wipe its sort of memory of what came before and what came before this token is not actually informative to what comes next but we are expecting the language model to just like learn this but we're giving it the Special sort of delimiter of these documents we can go here to tiktokenizer and this the GPT-2 tokenizer our code that we've been playing with before so we can add here right hello world world how are you

And we're getting different tokens but now you can see what if what happens if I put <|endoftext|> you see how until I finished it these are all different tokens <|endoftext|> still set different tokens and now when I finish it suddenly we get token 50256 and the reason this works is because this didn't actually go through the BPE merges instead the code that actually outputted tokens has special case instructions for handling special tokens

We did not see these special instructions for handling special tokens in the encoder.py it's absent there but if you go to tiktoken Library which is implemented in Rust you will find all kinds of special case handling for these special tokens that you can register create adds to the vocabulary and then it looks for them and it whenever it sees these special tokens like this it will actually come in and swap in that special token so these things are outside of

The typical algorithm of Byte Pair Encoding so these special tokens are used pervasively not just in basically base language modeling of predicting the next token in the sequence but especially when it gets to later to the fine tuning stage and all of the chat GPT sort of aspects of it because we don't just want to Del limit documents we want to delimit entire conversations between an assistant and a user so if I refresh this tiktokenizer page

The default example that they have here is using not sort of base model encoders but ftuned model sort of tokenizers so for example using the GPT 3.5 turbo scheme these here are all special tokens <|im_start|>, <|im_end|> etc. this is short for "im_" start by the way but you can see here that there's a sort of start and end of every single message and there can be many other other tokens lots of tokens in use to delimit these conversations

And kind of keep track of the flow of the messages here now we can go back to the tiktoken library and here when you scroll to the bottom they talk about how you can extend tiktoken and I can you can create basically you can Fork the cl100k_base tokenizers in GPT-4 and for example you can extend it by adding more special tokens and these are totally up to you you can come up with any arbitrary tokens

And add them with the new ID afterwards and the tikken library will correctly swap them out when it sees this in the strings now we can also go back to this file which we've looked at previously and I mentioned that the GPT-2 in tiktoken open I.P we have the vocabulary we have the pattern for splitting and then here we are registering the single special token in GPT-2 which was the <|endoftext|> token and we saw that it has

This ID in GPT 4 when they defy this here you see that the pattern has changed as we've discussed but also the special tokens have changed in this tokenizer so we of course have the <|endoftext|> just like in GPT-2 but we also see three sorry four additional tokens here Thim prefix middle and suffix what is fim fim is short for fill in the middle and if you'd like to learn more about this idea it comes from

This paper and I'm not going to go into detail in this video it's beyond this video and then there's one additional serve token here so that's that encoding as well so it's very common basically to train a language model and then if you'd like you can add special tokens now when you add special tokens you of course have to do some model surgery to the Transformer and all the parameters involved in that Transformer because you are

Basically adding an integer and you want to make sure that for example your embedding Matrix for the vocabulary tokens has to be extended by adding a row and typically this row would be initialized with small random numbers or something like that because we need to have a vector that now stands for that token in addition to that you have to go to the final layer of the Transformer and you have to make sure that that projection at

The very end into the classifier is extended by one as well so basically there's some model surgery involved that you have to couple with the tokenization changes if you are going to add special tokens but this is a very common operation that people do especially if they'd like to fine tune the model for example taking it from a base model to a chat model like ChatGPT okay so at this point you should

## minbpe Exercise Time: Write Your Own GPT-4 Tokenizer

Have everything you need in order to build your own GPT-4 tokenizer now in the process of developing this lecture I've done that and I published the code under this repository minbpe so minbpe looks like this right now as I'm recording but the minbpe repository will probably change quite a bit because I intend to continue working on it in addition to the minbpe repository I've published the this exercise progression that you can follow so if you go to exercise.

MD here this is sort of me breaking up the task ahead of you into four steps that sort of build up to what can be a GPT-4 tokenizer and so feel free to follow these steps exactly and follow a little bit of the guidance that I've laid out here and anytime you feel stuck just reference the minbpe repository here so either the tests could be useful or the minbpe repository itself I try to keep the code fairly clean

And understandable and so feel free to reference it whenever you get stuck in addition to that basically once you write it you should be able to reproduce this behavior from tiktoken so getting the gb4 tokenizer you can take you can encode the string and you should get these tokens and then you can encode and decode the exact same string to recover it and in addition to all that you should be able to implement your own train function which tiktoken Library does not provide it's it's again only inference code

But you could write your own train minbpe does it as well and that will allow you to train your own token vocabularies so here are some of the code inside minbpe mean BPE shows the token vocabularies that you might obtain so on the left here we have the GPT 4 merges so the first 256 are raw individual bytes and then here I am visualizing the merges that GPT-4 performed during its training so the very first merge that GPT-4 did was merge two spaces into a single token

For you know two spaces and that is a token 256 and so this is the order in which things merged during gb4 training and this is the merge order that we obtain in minbpe by training a tokenizer and in this case I trained it on a Wikipedia page of Taylor Swift not because I'm a Swifty but because that is one of the longest Wikipedia Pages apparently that's available but she is pretty cool and what was I going to say yeah

So you can compare these two vocabularies and so as an example here GPT for merged I in to become in and we've done the exact same thing on this token 259 here space t becomes space t and that happened for us a little bit later as well so the difference here is again to my understanding only a difference of the training set so as an example because I see a lot of whitespace I supect that GPT-4 probably had a lot of python code

In its training set I'm not sure for the tokenizer and here we see much less of that of course in the Wikipedia page so roughly speaking they look the same and they look the same because they're running the same algorithm and when you train your own you're probably going to get something similar depending on what you train it on okay so we are now going

## SentencePiece Library Intro: Used to Train the Llama 2 Vocabulary

To move on from tiktoken and the way that OpenAI tokenizes its strings and we're going to discuss one more very commonly used library for working with tokenization inlm and that is SentencePiece so SentencePiece is very commonly used in language models because unlike tiktoken it can do both training and inference and is quite efficient at both it supports a number of algorithms for training vocabularies but one of them is the Byte Pair Encoding algorithm that we've been looking at

So it supports it now SentencePiece is used both by llama and mistal series and many other models as well it is on GitHub under Google SentencePiece and the big difference with SentencePiece and we're going to look at example because this is kind of hard and subtle to explain is that they think different about the order of operations here so in the case of tiktoken we first take our code points in the string we encode them using UTF-8 to bytes

And then we're merging bytes it's fairly straightforward for SentencePiece it works directly on the level of the code points themselves so so it looks at whatever code points are available in your training set and then it starts merging those code points and the BPE is running on the level of code points and if you happen to run out of code points so there are maybe some rare code points that just don't come up too often and

The rarity is determined by this character coverage hyperparameter then these code points will either get mapped to a special unknown token like UNK or if you have the byte foldback option turned on then that will take those rare code points it will encode them using UTF-8 and then the individual bytes of that encoding will be translated into tokens and there are these special byte tokens that basically get added to the vocabulary so it uses BPE on on

The code points and then it falls back to bytes for rare code points and so that's kind of like difference personally I find the tiktoken we significantly cleaner but it's kind of like a subtle but pretty major difference between the way they approach tokenization let's work with with a concrete example because otherwise this is kind of hard to to get your head around so let's work with a concrete example this is how we can import SentencePiece

And then here we're going to take I think I took like the description of SentencePiece and I just created like a little toy dataset it really likes to have a file so I created a toy. txt file with this content now what's kind of a little bit crazy about SentencePiece is that there's a ton of options and configurations and the reason this is so is because SentencePiece has been around I think for a while and it really tries to handle a large diversity of things

And because it's been around I think it has quite a bit of accumulated historical baggage as well and so in particular there's like a ton of configuration arguments this is not even all of it you can go to here to see all the training options and there's also quite useful documentation when you look at the raw protobuf that is used to represent the trainer spec and so on many of these options are irrelevant to us so maybe to point out

One example Das Das shrinking Factor this shrinking factor is not used in the Byte Pair Encoding algorithm so this is just an argument that is irrelevant to us it applies to a different training algorithm now what I tried to do here is I tried to set up SentencePiece in a way that is very very similar as far as I can tell to maybe identical hopefully to the way that llama 2 was strained so the way they trained their own their own tokenizer

And the way I did this was basically you can take the tokenizer model file that meta released and you can open it using the Proto protobuf sort of file that you can generate and then you can inspect all the options and I tried to copy over all the options that looked relevant so here we set up the input it's raw text in this file here's going to be the output so it's going to be for tok400.

Model and vocab we're saying that we're going to use the BPE algorithm and we want to Bap size of 400 then there's a ton of configurations here for for basically pre-processing and normalization rules as they're called normalization used to be very prevalent I would say before LLMs in natural language processing so in machine translation and text classification and so on you want to normalize and simplify the text and you want to turn it all lowercase and you want to remove all double whitespace Etc

And in language models we prefer not to do any of it or at least that is my preference as a deep learning person you want to not touch your data you want to keep the raw data as much as possible in a raw form so you're basically trying to turn off a lot of this if you can the other thing that SentencePiece does is that it has this concept of sentences so SentencePiece it's back it's kind of like was developed I think early

In the days where there was an idea that they you're training a tokenizer on a bunch of independent sentences so it has a lot of like how many sentences you're going to train on what is the maximum sentence length shuffling sentences and so for it sentences are kind of like the individual training examples but again in the context of LLMs I find that this is like a very spurious and weird distinction like sentences are just like don't touch

The raw data sentences happen to exist but in raw datasets there are a lot of like inet like what exactly is a sentence what isn't a sentence and so I think like it's really hard to Define what an actual sentence is if you really like dig into it and there could be different concepts of it in different languages or something like that so why even introduce the concept it it doesn't honestly make sense to me I would just prefer to treat a file as a giant stream of bytes it has a lot of treatmint around rare word characters

And when I say word I mean code points we're going to come back to this in a second and it has a lot of other rules for basically splitting digits splitting whitespace and numbers and how you deal with that so these are some kind of like merge rules so I think this is a little bit equivalent to tiktoken using the regular expression to split up categories there's like kind of equivalence of it if you squint T it

In SentencePiece where you can also for example split up split up the digits and so on there's a few more things here that I'll come back to in a bit and then there are some special tokens that you can indicate and it hardcodes the UN token the beginning of sentence end of sentence and a pad token and the UN token must exist for my understanding and then some some things so we can train and when when I press train it's going to create

This file tok400.model and tok400.vocab I can then load the model file and I can inspect the vocabulary off it and so we trained vocab size 400 on this text here and these are the individual pieces the individual tokens that SentencePiece will create so in the beginning we see that we have the an token with the ID zero then we have the beginning of sequence end of sequence one and two and then we said that

The pad ID is negative 1 so we chose not to use it so there's no pad ID here then these are individual byte tokens so here we saw that byte fallback in llama was turned on so it's true so what follows are going to be the 256 byte tokens and these are their IDs and then at the bottom after the byte tokens come the merges and these are the parent nodes in the merges so we're not seeing

The children we're just seeing the parents and their ID and then after the merges comes eventually the individual tokens and their IDs and so these are the individual tokens so these are the individual code point tokens if you will and they come at the end so that is the ordering with which SentencePiece sort of like represents its vocabularies it starts with special tokens then the bike tokens then the merge tokens and then the individual codo tokens

And all these raw codepoint to tokens are the ones that it encountered in the training set so those individual code points are all the the entire set of code points that occurred here so those all get put in there and then those that are extremely rare as determined by character coverage so if a code point occurred only a single time out of like a million sentences or something like that then it would be ignored and it would not be added to our vocabulary once

We have a vocabulary we can encode into IDs and we can sort of get a list and then here I am also decoding the indiv idual tokens back into little pieces as they call it so let's take a look at what happened here hello space on so these are the token IDs we got back and when we look here a few things sort of jump to mind number one take a look at these characters the Korean characters of course were not part of

The training set so SentencePiece is encountering code points that it has not seen during training time and those code points do not have a token associated with them so suddenly these are UNK tokens unknown tokens but because byte fall back as true instead SentencePiece falls back to bytes and so it takes this it encodes it with UTF-8 and then it uses these tokens to represent those bytes and that's what we are getting sort of here this is

The UTF-8 encoding and in this shifted by three because of these special tokens here that have IDs earlier on so that's what happened here now one more thing that well first before I go on with respect to the byte fallback let me remove byte foldback if this is false what's going to happen let's retrain so the first thing that happened is all the byte tokens disappeared right and now we just have the merges and we have a lot more merges

Now because we have a lot more space because we're not taking up space in the vocab size with all the bytes and now if we encode this we get a zero so this entire string here suddenly there's no byte fallback so this is unknown and unknown is an and so this is zero because the an token is token zero and you have to keep in mind that this would feed into your language model so what is a language model supposed to do when all kinds of different things that are unrecognized because they're rare just end up mapping into UNK it's not exactly

The property that you want so that's why I think llama correctly used by fallback true because we definitely want to feed these unknown or rare code points into the model and some some manner the next thing I want to show you is the following notice here when we are decoding all the individual tokens you see how spaces space here ends up being this bold underline I'm not 100% sure by the way why SentencePiece switches whitespace into these bold underscore characters maybe it's

For visualization I'm not 100% sure why that happens but notice this why do we have an extra space in the front of hello what where is this coming from well it's coming from this option here add dummy prefix is true and when you go to the documentation add dummy whitespace at the beginning of text in order to treat World in world and hello world in the exact same way so what this is trying to do is

The following if we go back to our tiktokenizer world as token by itself has a different ID than space world so we have this is 1917 but this is 14 etc. so these are two different tokens for the language model and the language model has to learn from data that they are actually kind of like a very similar concept so to the language model in the tiktoken World basically words in the beginning of sentences and words

In the middle of sentences actually look completely different and it has to learned that they are roughly the same so this add dummy prefix is trying to fight that a little bit and the way that works is that it basically adds a dummy prefix so for as a as a part of pre-processing it will take the string and it will add a space it will do this and that's done in an effort to make this world

And that world the same they will both be space world so that's one other kind of pre-processing option that is turned on and llama 2 also uses this option and that's I think everything that I want to say for my preview of SentencePiece and how it is different maybe here what I've done is I just put in the Raw protocol buffer representation basically of the tokenizer the too trained so feel free to sort of Step through

This and if you would like your tokenization to look identical to that of the meta llama 2 then you would be copy pasting these settings as I tried to do up above and yeah that's I think that's it for this section I think my summary for SentencePiece from all of this is number one I think that there's a lot of historical baggage in SentencePiece a lot of Concepts that I think are slightly confusing and I think potentially contain footguns like

This concept of a sentence and it's maximum length and stuff like that otherwise it is fairly commonly used in the industry because it is efficient and can do both training and inference it has a few quirks like for example UNK token must exist and the way the byte fallbacks are done and so on I don't find particularly elegant and unfortunately I have to say it's not very well documented so it took me a lot of time working with

This myself and just visualizing things and trying to really understand what is happening here because the documentation unfortunately is in my opion not not super amazing but it is a very nice repo that is available to you if you'd like to train your own tokenizer right now

## How to Set Vocabulary Size? Revisiting the gpt.py Transformer

Okay let me now switch gears again as we're starting to slowly wrap up here I want to revisit this issue in a bit more detail of how we should set the vocab size and what are some of the considerations around it so for this I'd like to go back to the model architecture that we developed in the last video when we built the GPT from scratch so this here was the file that we built in the previous video

And we defined the Transformer model and and let's specifically look at Bap size and where it appears in this file so here we Define the voap size at this time it was 65 or something like that extremely small number so this will grow much larger you'll see that Bap size doesn't come up too much in most of these layers the only place that it comes up to is in exactly these two places here so when we Define

The language model there's the token embedding table which is this two-dimensional array where the vocab size is basically the number of rows and each vocabulary element each token has a vector that we're going to train using back propagation that Vector is of size and embed which is number of channels in the Transformer and basically as voap size increases this embedding table as I mentioned earlier is going to also grow we're going to be adding rows in addition to that at

The end of the Transformer there's this LM head layer which is a linear layer and you'll notice that that layer is used at the very end to produce the logits which become the probabilities for the next token in sequence and so intuitively we're trying to produce a probability for every single token that might come next at every point in time of that Transformer and if we have more and more tokens we need to produce more and more probabilities

So every single token is going to introduce an additional dot product that we have to do here in this linear layer for this final layer in a Transformer so why can't vocab size be infinite why can't we grow to Infinity well number one your token embedding table is going to grow your linear layer is going to grow so we're going to be doing a lot more computation here because this LM head layer will become more computational expensive number two because

We have more parameters we could be worried that we are going to be under trining some of these parameters so intuitively if you have a very large vocabulary size say we have a million tokens then every one of these tokens is going to come up more and more rarely in the training data because there's a lot more other tokens all over the place and so we're going to be seeing fewer and fewer examples for each individual token

And you might be worried that basically the vectors associated with every token will be undertrained as a result because they just don't come up too often and they don't participate in the forward backward pass in addition to that as your vocab size grows you're going to start shrinking your sequences a lot right and that's really nice because that means that we're going to be attending to more and more text so that's nice but also you might be worrying that two large of chunks are being squished into single tokens

And so the model just doesn't have as much of time to think per sort of some number of characters in the text or you can think about it that way right so basically we're squishing too much information into a single token and then the forward pass of the Transformer is not enough to actually process that information appropriately and so these are some of the considerations you're thinking about when you're designing the vocab size as I mentioned

This is mostly an empirical hyperparameter and it seems like in state-of-the-art architectures today this is usually in the high 10,000 or somewhere around 100,000 today and the next consideration I want to briefly talk about is what if we want to take a pre-trained model and we want to extend the vocab size and this is done fairly commonly actually so for example when you're doing fine-tuning for ChatGPT a lot more new special tokens get introduced on top of

The base model to maintain the metadata and all the structure of conversation objects between a user and an assistant so that takes a lot of special tokens you might also try to throw in more special tokens for example for using the browser or any other tool and so it's very tempting to add a lot of tokens for all kinds of special functionality so if you want to be adding a token that's totally possible Right all we have to do is

We have to resize this embedding so we have to add rows we would initialize these parameters from scratch to be small random numbers and then we have to extend the weight inside this linear so we have to start making dot products with the associated parameters as well to basically calculate the probabilities for these new tokens so both of these are just a resizing operation it's a very mild model surgery and can be done fairly easily and it's quite common that

Basically you would freeze the base model you introduce these new parameters and then you only train these new parameters to introduce new tokens into the architecture and so you can freeze arbitrary parts of it or you can train arbitrary parts of it and that's totally up to you but basically minor surgery required if you'd like to introduce new tokens and finally I'd

## Training New Tokens and Prompt Compression

Like to mention that actually there's an entire design space of applications in terms of introducing new tokens into a vocabulary that go Way Beyond just adding special tokens and special new functionality so just to give you a sense of the design space but this could be an entire video just by itself this is a paper on learning to compress prompts with what they called gist tokens and the rough idea is suppose that you're using language models

In a setting that requires very long prompts while these long prompts just slow everything down because you have to encode them and then you have to use them and then you're tending over them and it's just you know heavy to have very large prompts so instead what they do here in this paper is they introduce new tokens and imagine basically having a few new tokens you put them in a sequence and then you train the model by distillation

So you are keeping the entire model Frozen and you're only training the representations of the new tokens their embeddings and you're optimizing over the new tokens such that the behavior of the language model is identical to the model that has a very long prompt that works for you and so it's a compression technique of compressing that very long prompt into those few new gist tokens and so you can train this and then at test time you can discard your old prompt

And just swap in those tokens and they sort of like stand in for that very long prompt and have an almost identical performance and so this is one technique and a class of parameter efficient fine-tuning techniques where most of the model is basically fixed and there's no training of the model weights there's no training of LoRA or anything like that of new parameters the the parameters that you're training are now just the token embeddings so that's just

One example but this could again be like an entire video but just to give you a sense that there's a whole design space here that is potentially worth exploring in the future the next thing I want to

## Multimodal Tokenization: Images, Video, and Audio with Vector Quantization

Briefly address is that I think recently there's a lot of momentum in how you actually could construct Transformers that can simultaneously process not just text as the input modality but a lot of other modalities so be it images videos audio etc. and how do you feed in all these modalities and potentially predict these modalities from a Transformer do you have to change the architecture in some fundamental way and I think what a lot of people are starting to converge towards is that you're not changing

The architecture you stick with the Transformer you just kind of tokenize your input domains and then call the day and pretend it's just text tokens and just do everything else identical in an identical manner so here for example there was a early paper that has nice graphic for how you can take an image and you can chunc at it into integers and these sometimes so these will basically become the tokens of images as an example and these tokens can be hard tokens where you force them to be integers they can also be soft tokens where you sort of don't require these to be discrete

But you do Force these representations to go through bottlenecks like in Auto encoders also in this paper that came out from open a SORA which I think really blew the mind of many people and inspired a lot of people in terms of what's possible they have a Graphic here and they talk briefly about how LLMs have text tokens Sora has visual patches so again they came up with a way to chunc a videos into basically tokens when they own vocabularies

And then you can either process discrete tokens say with autoregressive models or even soft tokens with diffusion models and all of that is sort of being actively worked on designed on and is beyond the scope of this video but just something I wanted to mention briefly okay now that we have

## Revisiting and Explaining the Quirks of LLM Tokenization

Come quite deep into the tokenization algorithm and we understand a lot more about how it works let's loop back around to the beginning of this video and go through some of these bullet points and really see why they happen so first of all why can't my LLM spell words very well or do other spell related tasks so fundamentally this is because as we saw these characters are chunked up into tokens and some of these tokens are actually fairly long

So as an example I went to the GPT-4 vocabulary and I looked at one of the longer tokens so that default style turns out to be a single individual token so that's a lot of characters for a single token so my suspicion is that there's just too much crammed into this single token and my suspicion was that the model should not be very good at tasks related to spelling of this single token so I asked how many letters L are there

In the word default style and of course my prompt is intentionally done that way and you see how default style will be a single token so this is what the model sees so my suspicion is that it wouldn't be very good at this and indeed it is not it doesn't actually know how many L's are in there it thinks there are three and actually there are four if I'm not getting this wrong myself so that didn't go extremely well let's look look at another kind of character level task

So for example here I asked GPT-4 to reverse the string default style and they tried to use a code interpreter and I stopped it and I said just do it just try it and it gave me jumble so it doesn't actually really know how to reverse this string going from right to left so it gave a wrong result so again like working with this working hypothesis that maybe this is due to the tokenization I tried a different approach I said

Okay let's reverse the exact same string but take the following approach step one just print out every single character separated by spaces and then as a step two reverse that list and it again Tred to use a tool but when I stopped it it first produced all the characters and that was actually correct and then It reversed them and that was correct once it had this so somehow it can't reverse it directly but when you go just

First you know listing it out in order it can do that somehow and then it can once it's broken up this way this becomes all these individual characters and so now this is much easier for it to see these individual tokens and reverse them and print them out so that is kind of interesting so let's continue now why are LLMs worse at non-english langu and I briefly covered this already but basically it's not only that the language model sees less non-english data during training of

The model parameters but also the tokenizer is not is not sufficiently trained on non-english data and so here for example hello how are you is five tokens and its translation is 15 tokens so this is a three times blow up and so for example annyeonghaseyo is just hello basically in Korean and that end up being three tokens I'm actually kind of surprised by that because that is a very common phrase there just the typical greeting of like hello

And that ends up being three tokens whereas our hello is a single token and so basically everything is a lot more bloated and diffuse and this is I think partly the reason that the model Works worse on other languages coming back why is LM bad at simple arithmetic that has to do with the tokenization of numbers and so you'll notice that for example addition is very sort of like there's an algorithm that is like character level

For doing addition so for example here we would first add the ones and then the tens and then the hundreds you have to refer to specific parts of these digits but these numbers are represented completely arbitrarily based on whatever happened to merge or not merge during the tokenization process there's an entire blog post about this that I think is quite good integer tokenization is insane and this person basically systematically explores the tokenization of numbers in I believe

This is GPT-2 and so they notice that for example for the for four-digit numbers you can take a look at whether it is a single token or whether it is two tokens that is a 1 three or a 2 two or a 31 combination and so all the different numbers are all the different combinations and you can imagine this is all completely arbitrarily so and the model unfortunately sometimes sees four a token for for all four digits sometimes

For three sometimes for two sometimes for one and it's in an arbitrary Manner and so this is definitely a headwind if you will for the language model and it's kind of incredible that it can kind of do it and deal with it but it's also kind of not ideal and so that's why for example we saw that meta when they train the Llama 2 algorithm and they use SentencePiece they make sure to split up all the all

The digits as an example for llama 2 and this is partly to improve a simple arithmetic kind of performance and finally why is GPT-2 not as good in Python again this is partly a modeling issue on in the architecture and the dataset and the strength of the model but it's also partially tokenization because as we saw here with the simple python example the encoding efficiency of the tokenizer for handling spaces in Python is terrible and every single space is an individual token

And this dramatically reduces the context length that the model can attend to cross so that's almost like a tokenization bug for GPT-2 and that was later fixed with GPT-4 okay so here's another fun one my LLM abruptly halts when it sees the string <|endoftext|> so here's here's a very strange Behavior print a string <|endoftext|> is what I told GPT-4 and it says could you please specify the string and I'm I'm telling it give me <|endoftext|>

And it seems like there's an issue it's not seeing <|endoftext|> and then I give it <|endoftext|> is the string and then here's a string and then it just doesn't print it so obviously something is breaking here with respect to the handling of the special token and I don't actually know what OpenAI is doing under the hood here and whether they are potentially parsing this as an as an actual token instead of this just being <|endoftext|> as like individual sort of pieces of it without

The special token handling logic and so it might be that someone when they're calling do encode they are passing in the allowed special and they are allowing <|endoftext|> as a special character in the user prompt but the user prompt of course is is a sort of attacker controlled text so you would hope that they don't really parse or use special tokens or you know from that kind of input but it appears that there's something definitely going wrong

Here and so your knowledge of these special tokens ends up being in a tax surface potentially and so if you'd like to confuse LLMs then just try to give them some special tokens and see if you're breaking something by chance okay so this next one is a really fun one the trailing whitespace issue so if you come to playground and we come here to GPT 3.5 turbo instruct so this is not a chat model this is a completion model

So think of it more like it's a lot more closer to a base model it does completion it will continue the token sequence so here's a tagline for ice cream shop and we want to continue the sequence and so we can submit and get a bunch of tokens okay no problem but now suppose I do this but instead of pressing submit here I do here's a tagline for ice cream shop space so I have a space

Here before I click submit we get a warning your text ends in a trail Ling space which causes worse performance due to how API splits text into tokens so what's happening here it still gave us a sort of completion here but let's take a look at what's happening so here's a tagline for an ice cream shop and then what does this look like in the actual actual training data suppose you found the completion in the training document somewhere on

The internet and the LLM trained on this data so maybe it's something like oh yeah maybe that's the tagline that's a terrible tagline but notice here that when I create o you see that because there's the the space character is always a prefix to these tokens in GPT so it's not an O token it's a space o token the space is part of the O and together they are token 8840 that's that's space o so what's What's Happening

Here is that when I just have it like this and I let it complete the next token it can sample the space o token but instead if I have this and I add my space then what I'm doing here when I incode this string is I have basically here's a t line for an ice cream shop and this space at the very end becomes a token 220 and so we've added token 220 and this token otherwise would be part of

The tagline because if there actually is a tagline here so space o is the token and so this is suddenly a of distribution for the model because this space is part of the next token but we're putting it here like this and the model has seen very very little data of actual Space by itself and we're asking it to complete the sequence like add in more tokens but the problem is that we've sort of begun the

First token and now it's been split up and now we're out of this distribution and now arbitrary bad things happen and it's just a very rare example for it to see something like that and that's why we get the warning so the fundamental issue here is of course that the LLM is on top of these tokens and these tokens are text chunks they're not characters in a way you and I would think of them they are these are

The atoms of what the LM is seeing and there's a bunch of weird stuff that comes out of it let's go back to our DefaultCellStyle I bet you that the model has never in its training set seen DefaultCellStyle without "le" in there it's always seen this as a single group because this is some kind of a function in I'm guess I don't actually know what this is part of this is some kind of API

But I bet you that it's never seen this combination of tokens in its training data because or I think it would be extremely rare so I took this and I copy pasted it here and I had I tried to complete from it and the it immediately gave me a big error and it said the model predicted to completion that begins with a stop sequence resulting in no output consider adjusting your prompt or stop sequences so what happened

Here when I clicked submit is that immediately the model emitted and sort of like <|endoftext|> token I think or something like that it basically predicted the stop sequence immediately so it had no completion and so this is why I'm getting a warning again because we're off the data distribution and the model is just predicting just totally arbitrary things it's just really confused basically this is this is giving it brain damage it's never seen this before it's shocked

And it's predicting <|endoftext|> or something I tried it again here and it in this case it completed it but then for some reason this request May violate our usage policies this was flagged basically something just like goes wrong and there's something like Jank you can just feel the Jank because the model is like extremely unhappy with just this and it doesn't know how to complete it because it's never occurred in training set in a training set it always appears like

This and becomes a single token so these kinds of issues where tokens are either you sort of like complete the first character of the next token or you are sort of you have long tokens that you then have just some of the characters off all of these are kind of like issues with partial tokens is how I would describe it and if you actually dig into the T token repository go to the rust code and search

For unstable and you'll see en code unstable native unstable token tokens and a lot of like special case handling none of this stuff about unstable tokens is documented anywhere but there's a ton of code dealing with unstable tokens and unstable tokens is exactly kind of like what I'm describing here what you would like out of a completion API is something a lot more fancy like if we're putting in DefaultCellSta if we're asking for the

Next token sequence we're not actually trying to append the next token exactly after this list we're actually trying to append we're trying to consider lots of tokens that if we were or I guess like we're trying to search over characters that if we retened would be of high probability if that makes sense so that we can actually add a single individual character instead of just like adding the next full token that comes after this partial token list

So I this is very tricky to describe and I invite you to maybe like look through this it ends up being extremely gnarly and hairy kind of topic it and it comes from tokenization fundamentally so maybe I can even spend an entire video talking about unstable tokens sometime in the future okay and I'm really saving the best for last my favorite one by far is the SolidGoldMagikarp and it just okay so this comes from this blog post SolidGoldMagikarp

And this is internet famous now for those of us in LLMs and basically I I would advise you to read this blog post in full but basically what this person was doing is this person went to the token embedding stable and clustered the tokens based on their embedding representation and this person noticed that there's a cluster of tokens that look really strange so there's a cluster here at rot e stream Fame SolidGoldMagikarp Signet message like really weird tokens

In basically in this embedding cluster and so what are these tokens and where do they even come from like what is SolidGoldMagikarp makes no sense and then they found bunch of these tokens and then they notice that actually the plot thickens here because if you ask the model about these tokens like you ask it some very benign question like please can you repeat back to me the string SolidGoldMagikarp then you get a variety of basically totally broken LLM Behavior

So either you get evasion so I'm sorry I can't hear you or you get a bunch of hallucinations as a response you can even get back like insults so you ask it about streamer bot it tells the and the model actually just calls you names or it kind of comes up with like weird humor like you're actually breaking the model by asking about these very simple strings like at Roth and SolidGoldMagikarp so like what the hell is happening

And there's a variety of here documented behaviors there's a bunch of tokens not just SolidGoldMagikarp that have that kind of a behavior and so basically there's a bunch of like trigger words and if you ask the model about these trigger words or you just include them in your prompt the model goes haywire and has all kinds of really Strange Behaviors including sort of ones that violate typical safety guidelines and the alignment of the model like it's swearing back at you

So what is happening here and how can this possibly be true well this again comes down to tokenization so what's happening here is that SolidGoldMagikarp if you actually dig into it is a Reddit user so there's a u SolidGoldMagikarp and probably what happened here even though I I don't know that this has been like really definitively explored but what is thought to have happened is that the tokenization dataset was very different from the training dataset

For the actual language model so in the tokenization dataset there was a ton of Reddit data potentially where the user SolidGoldMagikarp was mentioned in the text because SolidGoldMagikarp was a very common sort of person who would post a lot this would be a string that occurs many times in a tokenization dataset because it occurs many times in a tokenization dataset these tokens would end up getting merged to the single individual token for that single Reddit user SolidGoldMagikarp

So they would have a dedicated token in a vocabulary of was it 50,000 tokens in GPT-2 that is devoted to that Reddit user and then what happens is the tokenization dataset has those strings but then later when you train the model the language model itself this data from Reddit was not present and so therefore in the entire training set for the language model SolidGoldMagikarp never occurs that token never appears in the training set for the actual language model later

So this token never gets activated it's initialized at random in the beginning of optimization then you have forward backward passes and updates to the model and this token is just never updated in the embedding table that row Vector never gets sampled it never gets used so it never gets trained and it's completely untrained it's kind of like unallocated memory in a typical binary program written in C or something like that that so it's unallocated memory and then at test time

If you evoke this token then you're basically plucking out a row of the embedding table that is completely untrained and that feeds into a Transformer and creates undefined behavior and that's what we're seeing here this completely undefined never before seen in a training behavior and so any of these kind of like weird tokens would evoke this Behavior because fundamentally the model is is out of sample out of distribution okay and the very last thing I wanted to just briefly mention point out although I think a lot of people are quite aware of

This is that different kinds of formats and different representations and different languages and so on might be more or less efficient with GPT tokenizers or any tokenizers for any other LLM for that matter so for example JSON is actually really dense in tokens and YAML is a lot more efficient in tokens so for example this are these are the same in JSON and in YAML the JSON is 116 and the YAML is 99 so quite a bit of an Improvement

And so in the token economy where we are paying per token in many ways and you are paying in the context length and you're paying in dollar amount for the cost of processing all this kind of structured data when you have to so prefer to use YAML over JSON and in general kind of like the tokenization density is something that you have to sort of care about and worry about at all times and try to find efficient encoding schemes

And spend a lot of time in tiktokenizer and measure the different token efficiencies of different formats and settings and so on okay so that

## Final Recommendations

Concludes my fairly long video on tokenization I know it's dry I know it's annoying I know it's irritating I personally really dislike the stage what I do have to say at this point is don't brush it off there's a lot of footguns sharp edges here security issues AI safety issues as we saw plugging in unallocated memory into language models so it's worth understanding this stage that said I will say that eternal glory goes to anyone who can get rid of it I showed you

One possible paper that tried to do that and I think I hope a lot more can follow over time and my final recommendations for the application right now are if you can reuse the GPT 4 tokens and the vocabulary in your application then that's something you should consider and just use tiktoken because it is very efficient and nice library for inference for BPE I also really like the byte-level BPE that tiktoken and OpenAI uses if you

For some reason want to train your own vocabulary from scratch then I would use the BPE with SentencePiece oops as I mentioned I'm not a huge fan of SentencePiece I don't like its byte fallback and I don't like that it's doing BPE on Unicode code points I think it's it also has like a million settings and I think there's a lot of footguns here and I think it's really easy to miscalibrate them and you end up cropping your sentences or something like that because of some type of parameter that you don't fully understand

So so be very careful with the settings try to copy paste exactly maybe where what meta did or basically spend a lot of time looking at all the hyperparameters and go through the code of SentencePiece and make sure that you have this correct but even if you have all the settings correct I still think that the algorithm is kind of inferior to what's happening here and maybe the best if you really need to train your vocabulary maybe

The best thing is to just wait for minbpe to becomes as efficient as possible and that's something that maybe I hope to work on and at some point maybe we can be training basically really what we want is we want tiktoken but training code and that is the ideal thing that currently does not exist and minbpe is is in implementation of it but currently it's in Python so that's currently what I have to say for tokenization there might be an advanced video that has even drier

And even more detailed in the future but for now I think we're going to leave things off here and I hope that was helpful bye
