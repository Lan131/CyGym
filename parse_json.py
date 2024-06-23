import pandas as pd
import json

# Path to your JSON file
json_file_path = '/Users/michaellanier/Desktop/Cyber_Defense_Simulator-main/useful_filtered_os_nvd_entries.json'

# Load the JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Parse the JSON data into a DataFrame
records = []
for item in data:
    CVE_id = item['CVE_id']
    source_identifier = item['source_identifier']
    published_time = item['published_time']
    lastModified_time = item['lastModified_time']
    cvss_metrics = item.get('cvss_metrics', {})
    baseScore = cvss_metrics.get('baseScore', None)
    baseSeverity = cvss_metrics.get('baseSeverity', None)
    exploitabilityScore = cvss_metrics.get('exploitabilityScore', None)
    impactScore = cvss_metrics.get('impactScore', None)

    for config in item['affected_configurations']:
        matchCriteriaId = config.get('matchCriteriaId', None)
        versionStartIncluding = config.get('versionStartIncluding', None)
        versionEndExcluding = config.get('versionEndExcluding', None)
        type_ = config.get('type', None)
        vendor = config.get('vendor', None)
        product = config.get('product', None)
        version = config.get('version', None)

        records.append({
            'CVE_id': CVE_id,
            'source_identifier': source_identifier,
            'published_time': published_time,
            'lastModified_time': lastModified_time,
            'baseScore': baseScore,
            'baseSeverity': baseSeverity,
            'exploitabilityScore': exploitabilityScore,
            'impactScore': impactScore,
            'matchCriteriaId': matchCriteriaId,
            'versionStartIncluding': versionStartIncluding,
            'versionEndExcluding': versionEndExcluding,
            'type': type_,
            'vendor': vendor,
            'product': product,
            'version': version
        })

# Convert records into a DataFrame
df = pd.DataFrame(records)

# Display the DataFrame
print(df)
df.to_csv("CVE.csv", index=False)
