---
name: Move Experiment
about: Move an existing experiment to Haystack Core or Integrations
title: ''
labels: ''
assignees: ''

---

**Steps for Moving an Experiment to Haystack Core or Integrations**
- [ ] Make sure the latest Haystack release or an integration release contains the merged experiment
- [ ] Update import statements in example cookbook, remove experimental tag from cookbook, etc.
- [ ] Close discussion in haystack-experimental with move information
- [ ] Remove pydocs
- [ ] Remove experiment from catalog in haystack-experimental README.md
- [ ] Remove example notebook from haystack-experimental if it exists
      
