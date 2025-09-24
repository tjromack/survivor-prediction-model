
# Season 49 Data Entry Instructions

## Overview
Fill out the season_49_template.csv file with contestant information as it becomes available.

## Data Entry Phases

### Phase 1: Pre-Season (Cast Announcement)
Fill out these columns immediately when cast is announced:
- **Contestant_Name**: Full name as shown on CBS
- **Age**: Age at time of filming
- **Gender**: M or F
- **Occupation**: Current job/profession
- **Occupation_Category**: Business, Healthcare, Education, Entertainment, Government, Sports, Military, Legal, Other
- **Home_State**: Two-letter state code (CA, TX, NY, etc.) or ON for Canada
- **Home_Region**: West, South, Midwest, Northeast, Canada
- **Relationship_Status**: Single, Married, In_Relationship, Divorced, Engaged
- **Athletic_Background**: None, Recreational, High_School, College, Professional
- **Physical_Build**: Small, Medium, Large (based on appearance)
- **Self_Reported_Fitness**: 1-5 scale (estimate from interviews)
- **Survivor_Knowledge**: No_Knowledge, Casual_Fan, Fan, Superfan (from interviews)
- **Strategic_Archetype**: Social_Player, Strategic_Player, Challenge_Beast, Under_Radar, Villain, Hero, Provider, Wild_Card
- **Pre_Game_Target_Size**: 1-5 scale (how big a target they seem pre-game)

### Phase 2: During Season (Update Weekly)
Update these as episodes air:
- **Tribal_Challenges_Won/Total**: Count team challenge wins
- **Individual_Challenges_Won/Total**: Count individual immunity wins
- **Advantages_Found/Played**: Hidden immunity idols, advantages
- **Alliance_Count**: Number of alliances formed
- **Votes_Against_Total**: Cumulative votes received
- **Tribals_Attended**: Number of tribal councils attended
- **Days_Lasted**: Update when eliminated
- **Made_Merge**: Y when merge happens, N when eliminated pre-merge

### Phase 3: Post-Season
Final data entry:
- **Final_Placement**: 1-18 placement
- **Elimination_Type**: Winner, Runner_Up, Fire_Challenge, Voted_Out, Medical
- **Jury_Votes_Received**: For finalists only
- **Made_Finale**: Y for final 3, N for others
- **Confessional_Count**: From episode transcripts
- **Screen_Time_Rank**: 1-18 ranking by screen time

## Value Guidelines

**Strategic_Archetype Options:**
- Social_Player: Focuses on relationships and social bonds
- Strategic_Player: Game-focused, strategic thinker
- Challenge_Beast: Physical competitor, challenge-focused
- Under_Radar: Quiet, low-threat gameplay
- Villain: Aggressive, confrontational style
- Hero: Likeable, moral compass type
- Provider: Camp life contributor
- Wild_Card: Unpredictable gameplay

**Occupation_Category Guidelines:**
- Business: Corporate, sales, marketing, finance, tech
- Healthcare: Doctors, nurses, therapists
- Education: Teachers, professors, administrators
- Entertainment: Actors, musicians, media
- Government: Public service, military, law enforcement
- Sports: Athletes, coaches, fitness trainers
- Military: Active or former military
- Legal: Lawyers, paralegals
- Other: Anything else

## Tips for Accuracy
1. Use official CBS cast bios for demographic info
2. Watch cast interviews for knowledge level and strategy
3. Update performance data after each episode
4. Double-check spelling of names and locations
5. Be consistent with categories across all contestants

## Model Integration
Once complete, the data will be processed through:
1. Feature engineering pipeline
2. Model prediction generation  
3. Accuracy tracking against actual outcomes
4. Model retraining if needed
        