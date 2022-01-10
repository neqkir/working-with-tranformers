from transformers import BartTokenizer, BartForConditionalGeneration
from IPython.display import display, Markdown

import torch
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

LONG_BORING_AI_ARTICLE = """
Some of the biggest advances coming in modern medicine may also be the tiniest.

For a few decades, scientists have been learning how to design molecules that can operate in the human body. We’re talking about tiny particles that can manipulate biological processes—designed to bind to and thus block the enzyme essential for HIV replication, to help destroy tumors but not healthy cells, to weld together arteries after surgery.

On this small a level, also known as the nanoscopic scale or nanoscale, chemists are building and manipulating matter as small as atoms, small molecules, proteins, antibodies, and DNA bases that are less than 100 nanometers wide. These objects are much smaller than the cells in your body. Many are even tinier than the whisper-thin membrane that holds a human cell together.
Nanoparticles’ size makes them particularly useful against foes such as cancer. Unlike normal blood vessels that branch smoothly and flow in the same direction, blood vessels in tumors are a disorderly mess. And just as grime builds up in bad plumbing, nanoparticles, it seems, can be designed to build up in these problem growths.

This quirk of tumors has led to a bewildering number of nanotech-related schemes to kill cancer. The classic approach involves stapling drugs
to nanoparticles that ensure delivery to tumors but not to healthy parts of the body. A wilder method involves things like using a special
kind of nanoparticle that absorbs infrared light. Shine a laser through the skin on the build-up, and the particles will heat up to fry the
tumor from the inside out.

Using lasers to cook tumors sounds like a great reason to throw chemotherapy out the window, except that it’s nearly impossible
to foresee all of the possible pitfalls of unleashing particles so small. There are countless different substances in each human cell,
and it’s very hard to check how a molecule would interact with every one of them. Chemists are good at predicting narrower outcomes,
such as how a particular molecule will work on its intended target, but they have had to resort to rough approximation to try to
predict all of the possible interactions new nanoscale creations would have throughout the body. For instance, one particle turned
out to be remarkably sticky. A whole blob of proteins stuck all over the particle like hair on a ball of tape, keeping it from even reaching its target.

There have been a number of success stories in which scientists were able design and release new molecules with confidence, as with the HIV drug Viracept. But the difficulty of computing the myriad of possible interactions has been the major limiting factor stalling widespread use. It’s not all sunshine and cancer cures. Nanoparticles have real risks.

Take nanotubes, tiny superstrong cylinders. Remember using lasers to cook tumors? That involves nanotubes.
Researchers also want to use nanotubes to regrow broken bones and connect brains with computers.
Although they hold a lot of promise, certain kinds of long carbon nanotubes are the same size and shape as asbestos.
Both are fibers thin enough to pierce deep inside the lungs but too long
for the immune system to engulf and destroy. Mouse experiments have already suggested that inhaling these nanotubes causes the
same harmful effects (such as a particularly deadly form of cancer) as the toxic mineral once widely used in building insulation.

Shorter nanotubes, however, don’t seem to be dangerous.
Nanoparticles can be built in all sorts of different ways, and it’s difficult to predict which ones will go bad.
Imagine if houses with three bathrooms gave everyone in them lung disease, while houses with two or four bathrooms were safe.
It gets to the central difficulty of these nanoscale creations—the unforeseen dangers that could be the difference between biomiracle and bioterror.

Enter machine learning, that big shiny promise to solve all of our complicated problems. The field holds a lot of potential when it
comes to handling questions where there are many possible right answers. Scientists often take inspiration from nature—evolution,
ant swarms, even our own brains—to teach machines the rules for making predictions and producing outcomes without explicitly giving
them step-by-step programming. Given the right inputs and guidelines, machines can be as good or even better than we are
at recognizing and acting on patterns and can do so even faster and on a larger scale than humans alone are capable of pulling off.

In earlier days, scanners could only recognize letters of specific fonts. Today, after feeding computers tens of thousands of
examples of handwritten digits to detect and extrapolate patterns from, ATMs are now reading handwriting on checks.
In nanotechnology, the research is similar: Just like many slightly different shapes can mean the same letter,
many slightly different molecules can mean the same effect. Setting up a computer to learn how different nanoparticles
might interact with the complex human body can assist with what were previously impossibly complex computations to predict billions of possible outcomes.
""".replace('\n','')


############# BART

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

article_input_ids = tokenizer.batch_encode_plus([LONG_BORING_AI_ARTICLE], return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(torch_device)

summary_ids = model.generate(article_input_ids,
                             num_beams=4,
                             length_penalty=2.0,
                             max_length=142,
                            # min_len=56,
                             no_repeat_ngram_size=3)

#decode the summary
summary_bart = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

#display the results

print('> **Summary written by BART for the long AI text: **' + '\n' + summary_bart)

print('\n' + '*'*100 + '\n')

###### Another BART example

TINY_TEXT = """
The researchers examined three types of coral in reefs off the
coast of Fiji ... The researchers found when fish were plentiful,
they would eat algae and seaweed off the corals, which appeared
to leave them more resistant to the bacterium Vibrio coralliilyticus, a bacterium associated with bleaching. The researchers suggested the algae, like warming temperatures, might render the
corals’ chemical defenses less effective, and the fish were protecting the coral by removing the algae.

""".replace('\n','')

article_input_ids = tokenizer.batch_encode_plus([TINY_TEXT], return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(torch_device)

summary_ids = model.generate(article_input_ids,
                             num_beams=4,
                             length_penalty=2.0,
                             max_length=142,
                            # min_len=56,
                             no_repeat_ngram_size=3)

#decode the summary
summary_bart_2 = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

#display the results

print('> **Summary written by BART for the short Fiji text: **' + '\n' + summary_bart_2)

print('\n' + '*'*100 + '\n')


############## GPT-2

# build gpt-2 model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2_tok = GPT2Tokenizer.from_pretrained('gpt2')

gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')#, output_past=True)

# truncate to 869 tokens so that we have space to generate another 155
enc = gpt2_tok.encode(LONG_BORING_AI_ARTICLE, max_length=1024-155, return_tensors='pt',truncation=True) 

# Generate another 155 tokens
source_and_summary_ids = gpt2_model.generate(enc, max_length=1024, do_sample=False)

# Only show the new ones
end_of_source = "An official statement said:" 

summary_gpt2 = gpt2_tok.decode(source_and_summary_ids[0]) 

print('> **Summary written by GPT-2: **' + '\n' + summary_gpt2)
