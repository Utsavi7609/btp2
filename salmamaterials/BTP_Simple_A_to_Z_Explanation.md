# BTP Simplified Story: From Start to Finish

This document is your cheat sheet. It is written in simple English so you can easily explain the entire BTP journey, step-by-step, to your professor during your thesis defense.

---

## 1. The Big Idea (What are we trying to prove?)

Imagine you show a person a sad movie clip. 
After the clip, you ask a survey: "Did you feel sad?" 
Because they don't want to look weak, they click "Neutral." They just lied on the survey. 

But can their body lie? No.

Their facial expression showed sadness. Their voice tone was sad. Most importantly, their heart rate changed. 
**Our goal is to prove "Affective Drift."** This means a person's conscious mind (what they click on a survey) often "drifts" away from the truth, but their physiological body (heart rate, face, voice) tells the objective truth. 

If we can prove this, we can build tools that use heart rate data to automatically play clips that make a user happy or calm them down, completely bypassing their stubborn conscious mind. 

## 2. The Four Key Letters (P, E, I, B)

To prove this, we tracked four things for every single video clip:
1. **$P$ (Perceived - The Brain):** What the user *thought* the video was about (e.g., "This video is a horror clip").
2. **$E$ (Expressed - The Movie Clip):** What the *video clip itself* actually portrayed. We looked at the clip's visual lighting/acting (Video) and the clip's music/dialogue (Audio).
3. **$I$ (Induced - The Survey):** What they *clicked on the survey* afterward (e.g., "I personally felt neutral").
4. **$B$ (Body - The Heart Rate):** What an AI guessed they felt by only looking at their Fitbit smartwatch data.

## 3. The Four Buckets (H1 to H4)

When we compare $P$, $E$, and $I$, the clips fall into buckets. You can explain these buckets as "types of clips":

*   **Bucket 1: The Honest Match (H1 Match).** 
    * *Example:* The movie clip is happy ($E$), the user thinks it's a happy video ($P$), and they honestly click "Happy" on the survey ($I$). 
    * *Use case:* Good for "Maintenance therapy" because the user transparently accepts the emotion.
*   **Bucket 2: The Feel Perception (H2 Feel).**
    * *Example:* They think it's a happy clip ($P$) and they click "Happy" ($I$), but the movie clip is actually totally neutral/boring ($E$). They projected their own mood onto the clip.
*   **Bucket 3: The Subconscious Trigger (H3 Drift).** 
    * *Example:* The movie clip is happy ($E$) and they know it's a happy clip ($P$), but they are stubborn and click "Neutral" on the survey ($I$).
    * *Use case:* Good for "Emotional Shifting therapy" because the stimulus was accurate, even if the brain denied it.
*   **Bucket 4: Chaos (H4 No Match).** 
    * *Example:* The video, their perception, and the survey are all totally different. Throw these clips out.

---

## 4. The Journey: How We Built the AI (The "How-To" Story)

### Step 1: Zero-Shot (Just Asking the AI)
First, we took the users' Fitbit heart rate numbers and fed them into an AI like Llama or Qwen. We used "Zero-Shot" prompting, which means we just gave the AI the numbers and asked: "What emotion is this?"
**The Result:** It failed. The AI's guesses were completely wrong (the Error score, or MAE, was over 1.1 on a 5-point scale). Why? Because Fitbit data is noisy and only updates once a minute.

### Step 2: Few-Shot (Teaching the AI)
Next, we decided to give the AI examples. We injected three medical examples describing how heart rate connects to emotion in the prompt. We called this "Few-Shot."
**The Result:** The AI got smarter at recognizing patterns, but the score on our Fitbit data was *still* bad (around 1.0 error). We were confused. Was the AI just dumb?

### Step 3: Checking a Clinical Dataset (K-EmoCon)
To test if the AI was dumb, we took the same code and ran it on a professional medical dataset called "K-EmoCon," because they use expensive hospital sensors that record 64 times a second.
**The Result:** The AI scored almost perfectly! It had an error rate of only 0.177 and was 100% accurate. 
**What this proved to your Professor:** The AI is brilliant. The math works. The problem had to be our Fitbit data.

### Step 4: Finding the "Recovery Paradox" Bug
We looked deeply at our Fitbit data. We realized two massive problems:
1.  **Lack of HRV:** Fitbits don't measure Heart Rate Variability well, so we mathematically injected a feature called **RMSSD** to give the AI stress clues.
2.  **The Timing Bug:** We realized our Python code was grabbing the heart rate data for the 2 minutes *after* the user clicked submit on the survey. This meant the AI was looking at the user's heart rate while they were resting and recovering from the video, NOT while they were actually shocked, scared, or happy watching it!

### Step 5: The "Shifted Window" Breakthrough
We solved the entire project with one fix. We shifted the time window 120 seconds backwards. Now, the AI was looking at the exact moment the person was watching the stimulus video on their screen.

## 5. The Final Run (Why only Qwen?)

We took our fixed, shifted dataset (1,495 rows) and ran it through **Qwen-32B**.
*   **Did we only use Qwen for the final run?** Yes. Why? Because we proved in the earlier steps that Llama, Llama-4, and Qwen all behave very similarly, but Qwen-32B is incredibly fast and mathematically consistent for this massive 1,495 clip run. You can tell your professor: "We used Llama-70B to prove the clinical benchmark, and Qwen-32B to execute the massive Fitbit data run for speed and efficiency."

**The Final Result:**
The result was incredible. The AI successfully processed 96.3% of the data. 
More importantly, our **H3 (Drift)** bucket exploded in size when we looked at the Fitbit (Body). It proved mathematically that a person's Fitbit ($B$) and the intentional movie clip ($E$) will often strongly agree on a feeling, while the person's conscious survey ($I$) denies it entirely. 

**Conclusion:** The project was a success. We successfully used AI to prove that biological signals can override subjective human surveys, unlocking new ways to recommend therapeutic media.
