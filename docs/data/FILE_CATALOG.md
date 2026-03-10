# Data File Catalog (MMLM 2026)

Canonical raw dataset root:
- `data/raw/march-machine-learning-mania-2026/`

## Core Identity Tables

- `MTeams.csv`, `WTeams.csv`: team master lists
  - key: `TeamID`
- `MSeasons.csv`, `WSeasons.csv`: season metadata
  - key: `Season`
- `Conferences.csv`: conference master list
  - key: `ConfAbbrev`
- `Cities.csv`: city master list
  - key: `CityID`

## Tournament Structure

- `MNCAATourneySeeds.csv`, `WNCAATourneySeeds.csv`
  - map `(Season, TeamID) -> Seed`
- `MNCAATourneySlots.csv`, `WNCAATourneySlots.csv`
  - bracket slot topology by season
- `MNCAATourneySeedRoundSlots.csv` (men only)
  - seed/round/slot mapping helper
- `data/tourney_round_lookup.csv` (project canonical)
  - normalized seed pair -> canonical round

## Game Results

- Regular season compact:
  - `MRegularSeasonCompactResults.csv`
  - `WRegularSeasonCompactResults.csv`
- Regular season detailed:
  - `MRegularSeasonDetailedResults.csv`
  - `WRegularSeasonDetailedResults.csv`
- NCAA tournament compact:
  - `MNCAATourneyCompactResults.csv`
  - `WNCAATourneyCompactResults.csv`
- NCAA tournament detailed:
  - `MNCAATourneyDetailedResults.csv`
  - `WNCAATourneyDetailedResults.csv`
- Secondary tournament compact:
  - `MSecondaryTourneyCompactResults.csv`
  - `WSecondaryTourneyCompactResults.csv`

Primary game join key:
- `Season, DayNum, WTeamID, LTeamID`

## Supplemental Tables

- `MGameCities.csv`, `WGameCities.csv`
  - game -> city mapping (`CityID`)
- `MConferenceTourneyGames.csv`, `WConferenceTourneyGames.csv`
  - identifies conference-tourney games
- `MTeamConferences.csv`, `WTeamConferences.csv`
  - team conference by season
- `MTeamCoaches.csv` (men)
  - coach intervals by season/day range
- `MMasseyOrdinals.csv` (men)
  - ranking systems by `Season`, `RankingDayNum`, `SystemName`, `TeamID`
- `MSecondaryTourneyTeams.csv`, `WSecondaryTourneyTeams.csv`
  - secondary-tournament participants

## Team Name Mapping

- Kaggle sources:
  - `MTeamSpellings.csv`
  - `WTeamSpellings.csv`
- Project canonical mapping:
  - `data/TeamSpellings.csv`
  - includes unified mapping + `espn_id`

## Submission Templates

- `SampleSubmissionStage1.csv`
- `SampleSubmissionStage2.csv`

Submission schema:
- `ID,Pred`
- `ID = Season_LowTeamID_HighTeamID`

## Related Policy Docs

- `docs/data/RELATIONSHIP_DIAGRAM.md`
- `docs/data/TOURNEY_ROUND_ASSIGNMENT.md`
- `docs/data/TEAM_SPELLINGS_POLICY.md`
