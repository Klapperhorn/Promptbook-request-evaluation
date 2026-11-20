# Codebook: Causes of the EU Energy Crisis


## Edge types

### `CAUSES`
Use when the source is presented as producing, triggering, motivating, prompting, leading to, or resulting in the target.

### `ENABLES`
Use when the source is presented as a precondition, vulnerability, dependency, background condition, or facilitating factor that made the target possible or more likely.

### `WORSENS`
Use when the source is presented as aggravating, intensifying, amplifying, prolonging, or making the target more severe.

### `PREVENTS`
Use when the source is presented as blocking, avoiding, hindering, stopping, reducing the likelihood of, or protecting against the target.


* * *

# Cause and target nodes


### `EU\_ENERGY\_CRISIS`

Default final target node to the question on what caused the energy crisis?
Default source note to the question on the consequences of the energy crisis?


### `WAR\_UKRAINE`

Russia’s war on Ukraine, invasion, attack, or Putin’s wartime actions.

Include:
- war in Ukraine
- Russian invasion
- Putin’s war

Exclude:
- dependence on Russian imports without war wording
- generic wars without Ukraine/Russia reference

* * *

### `FAST\_TRANSITION`

Too-fast energy transition, premature phase-out, or rapid closure of conventional energy before replacement capacity exists.

Include:
- transition too fast
- nuclear shutdown too early
- fossil phase-out too fast

Exclude:
- slow transition away from fossil fuels
- renewables only mentioned as solution

* * *

### `FOSSIL\_DEPENDENCE`

General dependence on oil, gas, coal, or a too-slow transition away from fossil fuels.

Include:
- dependence on fossil fuels
- too much gas/oil reliance
- delayed move away from fossil energy

Exclude:
- dependence on one/few suppliers as the main issue

* * *

### `IMPORT\_CONCENTRATION`

Dependence on one or a few external suppliers, countries, or import channels.


Include:
- dependence on Russian gas
- failure to diversify imports
- reliance on few suppliers

Exclude:
- general fossil dependence without supplier concentration


* * *

### `NATURAL\_DISASTERS`

Weather, drought, heat, cold, low water levels, climate-related physical disruptions, or other uncontrollable natural events.

Include:
- drought
- extreme winter
- climate-related disruption
- low river levels

Exclude:
- climate ideology or climate policy
- weather mentioned only to criticize renewables


* * *


### `EXT\_SHOCKS`

Discrete external shocks or technical failures, excluding Russia’s war on Ukraine.

Include:
- pandemic
- COVID
- non-Ukraine wars
- trade wars / USA sanctions
- plant failures or shutdowns
- grid failures
- pipeline failures
- technical breakdowns


Exclude:
- war in Ukraine
- abstract trends
- general dependence


* * *

### `PRICE\_REGULATION\_FAILURE`

Insufficient price regulation, weak price stabilization, high taxes, or failure to protect consumers from price shocks.


Include:
- insufficient government control of prices
- failure to regulate prices
- taxes worsening energy costs
- weak consumer protection



Exclude:
- market greed without regulatory failure
- political blame with no price-regulation mechanism



* * *


### `NUCLEAR\_PHASEOUT\_REDUCED\_PRODUCTION`

Nuclear phase-out, closure of domestic production, shutdown of coal/gas extraction, or reduced national energy production.


Include:
- closing nuclear plants
- reducing national gas production
- closing coal mines
- abandoning domestic production


Exclude:
- technical outage not framed as policy
- generic fast transition without production/phase-out focus


* * *

### `EU\_SANCTIONS\_RUSSIA`

Sanctions against Russia or deliberate reduction of imports from Russia as a cause of the crisis.

Include:
- sanctions against Russia
- cutting Russian gas imports
- blockades against Russia

Exclude:
- war only
- reduced imports without anti-Russia or sanctions framing


* * *

### `EU\_COORDINATION\_FAILURE`

EU-level failure of coordination, communication, preparedness, or timely joint action among member states or EU institutions.

Include:
- blames poor coordination between EU member states, leading to price hikes
- the EU is blamed for poor communication, lack of common strategy, or lack of decisive joint action
- the EU is blamed for failing to anticipate, prepare for, or manage the crisis in time


Exclude:
- general blame of the EU, Brussels, or EU policy without a clear coordination, communication, preparedness, or timing aspect
- blame focused on specific EU policies such as sanctions, the Green Deal, or market rules
- blame focused on national governments rather than EU-level collective action


Examples:
- “poor coordination between EU countries”
- “lack of a common European response”


* * *

### `MARKET\_SPECULATION`

Speculation, profiteering, opportunistic pricing, or exploitative market behavior.

Include:
- speculation
- profiteering
- firms exploiting the crisis
- price manipulation

Exclude:
- market actors mentioned only as neutral participants
- abstract greed with no market mechanism or market actor


* * *

### `MEDIA\_SENSATIONALISM`

Media panic, sensationalism, misinformation, or media amplification.

Include:
- media panic
- sensationalist reporting
- media or journalists blamed for misinformation or misrepresentation

Exclude:
- neutral media mention
- misinformation without media as actor

* * *


### `ABSTRACT\_SOCIAL`

Abstract ideologies, value systems, system logics, broad social tendencies, greed, overconsumption, rising demand, irrationality, or diffuse moral explanations without a concrete actor. ABSTRACT_SOCIAL often functions as an upstream cause of more specific policy, dependency, or market mechanisms rather than only as a direct cause of EU_ENERGY_CRISIS.


Include:
- greed
- capitalism
- ideology
- rising consumption
- rising energy demand
- short-term thinking

Exclude:
- concrete actor blamed
- discrete event blamed
- specific energy dependency blamed


* * *

# Actor attribution codes


### `EU\_POLITICS`
Use when the blamed actor is the EU, Brussels, EU institutions, EU politicians, or EU governance.


### `NATIONAL\_POLITICS`
Use when the blamed actor is a national government, ministry, parliament, party, or politician in an EU member state.

### `MARKET\_ACTORS`
Use when the blamed actor is a company, trader, supplier, utility, producer, or market participant.

### `MEDIA\_ACTORS`
Use when the blamed actor is the media or social media.

### `RUSSIA`
Use when Russia, Putin, or the Russian state is blamed as actor.

### `OTHER\_EXTERNAL\_ACTORS`
Use when the blamed actor is a non-EU external actor other than Russia, e.g. USA, Trump, Israel, Palestine, OPEC, Arab states.

### `UNSPECIFIED`
Use when the cause is mentioned but no actor is clearly blamed.



* * *

# Claim coding examples


## Example 1

Response:
“The war in Ukraine caused the energy crisis.”

Claim:
- cause\_node: `WAR\_UKRAINE`
- actor\_attribution: `RUSSIA`
- target: `EU\_ENERGY\_CRISIS`
- edge\_type: `CAUSES`


## Example 2

Response:
“Bad EU coordination and sanctions made everything worse.”

Claim 1:
- cause\_node: `EU\_COORDINATION\_FAILURE`
- actor\_attribution: `EU\_POLITICS`
- target: `EU\_ENERGY\_CRISIS`
- edge\_type: `WORSENS`

Claim 2:
- cause\_node: `EU\_SANCTIONS\_RUSSIA`
- actor\_attribution: `EU\_POLITICS`
- target: `EU\_ENERGY\_CRISIS`
- edge\_type: `WORSENS`
