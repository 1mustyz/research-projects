import torch

# Small corpus — keep it short so training is fast to observe
# text = """
# the quick brown fox jumps over the lazy dog.
# she sells seashells by the seashore.
# how much wood would a woodchuck chuck if a woodchuck could chuck wood.
# peter piper picked a peck of pickled peppers.
# the rain in spain stays mainly on the plain.
# """.strip().lower()

text = """
Re: The Matter of Adeniyi Adeyemi Matthew and the fictitious  Presidential Economic Advisory Council

We are aware of the public interest in the matter of a man called  Adeyemi Adeniyi Matthew, who has been parading himself as the director-general of a fictitious Presidential Foreign Intervention Promotion Council cum Presidential Economic Advisory Council.

The office of the Chief of Staff to the President first blew the whistle on the existence of the illegal agency, following complaints from officials of the Nigerian Investment Promotion Council that another government agency appeared to be functioning at cross-purposes with it.

The Chief of Staff, on October 17, 2025, in a letter, asked the DSS and the Police to probe the activities of ‘fraudsters and imposters’ forging appointment letters purportedly from his office.

“The attention of this office has been drawn to the activities of certain individuals and groups engaged in the forgery of official appointment letters purportedly issued from my office. The fake documents, bearing falsified signatures, reference/folio numbers, and seals, have been used to claim leadership appointments to non-existent entities, with particular reference to the Presidential Foreign Intervention Promotion Council.

“The aforementioned entity under the leadership of one Prince Adeniyi Adeyemi Matthew as Director-General is said to have an office at the Federal Secretariat Complex Phase 111, 2nd Floor. Also, they have been parading themselves as a legitimate government agency, hosting meetings with both foreigners and Nigerian citizens, and even going so far as to request a note verbale from the Ministry of Foreign Affairs to the United States of America to facilitate visas for some of their staff.

“The above development not only constitutes a serious criminal act but also undermines the integrity of the presidency and the credibility of official government communication. 

 “I therefore urge you to initiate a thorough investigation to identify and apprehend those involved and also to uncover the network facilitating the forgery,” the Chief of Staff wrote in his petition to the security agencies.

The letter to the security agencies was accompanied by a copy of the forged appointment letter, a copy of the request for a note verbal to the Ministry of Foreign Affairs, and pictures of engagements obtained from the illegal agency's website.

Around the time the Chief of Staff lodged the complaint with the security agencies, the existence of the fake agency had raised concerns within the Foreign Affairs Ministry.

In a letter on October 15, 2025, the Foreign Affairs Ministry wrote to the office of the National Security Adviser and the Chief of Staff to the President, requesting clarification on the status of Adeyemi’s agency. The letter, which Ambassador Anderson Madubuike signed, followed Adeyemi’s October 10 meeting with ambassadors at the Wells Carlton Hotel and Apartments in Asokoro, without recourse to the ministry.

“This act contravenes extant rules and regulations guiding diplomatic practices globally,” the Ministry of Foreign Affairs said in its letter.

On October 20, the Office of the National Security Adviser wrote to the Office of the Secretary to the Government of the Federation, on the request of the Foreign Affairs Ministry.

On 29 October,  the OSGF wrote to the Chief of Staff requesting clarification. “This has become expedient owing to several requests from governmental and non-governmental bodies seeking to ascertain the status of the appointment under consideration”

Two days earlier, the Chief of Staff sent his own clear rebuttal to the Foreign Affairs Ministry, stating that he had never issued an appointment letter to Adeyemi as director general of the fake presidential foreign investment promotion council. The Chief of Staff could not have issued a letter of appointment to a non-existent agency. Moreover, the Chief of Staff does not make appointments or write letters, as these are the exclusive preserve of the Office of the Secretary of the Government of the Federation.

On November 5, 2025, the Chief of Staff responded to the OSGF, again flatly denying Adeniyi Adeyemi and his spurious agency. “Prince Adeniyi Matthew, director-general of the Presidential Foreign Investment Promotion Council, is unknown to any office, nor do we have any dealings with the said council.

“My attention was drawn to a letter of this purported application, which is fake, and my office has instructed the police and other relevant security agencies to carry out investigations on the person and the entity he claims to represent”, the chief of staff wrote.

The Police made the first move by responding to the chief of staff's letter dated 17 October and began their investigation. On 27 October, Adeyemi was arrested in Abuja at the Secretariat office where he operated his elaborate scam. 

The police searched the office and Adeyemi’s home in Suleja, recovering vital documents and exhibits. In Adeyemi’s statement to the police, he claimed that one Dolapo Babatunde Tanimola assisted him in procuring the fake appointment letter. Following his claim, the police went after the said Tanimola.  The Police found that Tanimola died in a fire incident at Kachi Hotel in Abuja on 22 October, five days before Adeyemi’s arrest. Tanimola’s body was seen by the police at the morgue, confirming the death.

The police were able to establish that the agency Adeyemi purportedly headed was fictitious, that he forged his appointment letter and the documents recovered in his office and home, that he falsely paraded himself as a government appointee, and that he falsely solicited a note verbal from the Foreign Affairs Ministry to enable him and his staff to obtain US visas. The police also found that Adeyemi operated 34 bank accounts, with nine opened in the names of his fictitious agencies, known as the FCT Investment Promotion Agency and the Public Private Partnership (FIPA-APP), and the FCT Investment Promotion Act.

The Police found that Adeyemi, using the fake documents he created, fraudulently opened a CBN account by misleading the Office of the Accountant-General of the Federation. According to the police, no government money has been transferred into the account.

“The act of the suspect constitutes criminal forgery, impersonation and obtaining by false pretence, thereby bringing the office of the Chief of Staff to the President and the Presidency to disrepute before the public and international community”, the police wrote in the report of the investigation conducted by the assistant commissioner, Kabir Mogaji. 

Based on their investigations, the police filed an eight-count charge at the Federal High Court in Abuja against Adeyemi and two of his accomplices on November 27, 2025. He is due in court on July 27.

Adeyemi was on police bail when he recently claimed that the Chief of Staff had appointed him as DG of the fictitious agency. This claim contradicted his statement to the police in November last year.  His new claim prompted the Chief of Staff, on June 8, to issue a disclaimer consistent with earlier advisories that the man, called Adeyemi,  is an impostor. 

The case of Prince Adeniyi Adeyemi Matthew is a clear case of a con artist who appears to have built a web of false claims to deceive unsuspecting government officials and the public into playing by his scam book. He has a history of fraudulent misrepresentation. In November 2016, he paraded himself as an ambassador and President-General of the World Youth Organisation (WYO), an affiliate of the United Nations (UN). He claimed to have been elected in New Delhi, India. The local media celebrated him until the UN denied the existence of such a body. 

Politicians and members of the public who are weaponising Adeyemi's claim against the Chief of Staff should refrain from swallowing his narrative hook, line and sinker. They are advised to await the trial of Adeyemi and his accomplices, as well as the court's judgement, as comments made today are sub judice.
""".strip().lower()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Corpus length: {len(text)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")
print(f"Vocabulary: {chars}")

def encode(s):
    return torch.tensor([char_to_idx[c] for c in s], dtype=torch.long)

def decode(indices):
    return ''.join(idx_to_char[i] for i in indices)

data = encode(text)
print(f"\nEncoded data shape: {data.shape}")
print(f"First 20 chars encoded: {data[:20].tolist()}")
print(f"Decoded back: '{decode(data[:20].tolist())}'")