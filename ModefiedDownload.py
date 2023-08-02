from Bio import Entrez
import xml.etree.ElementTree as ET
import pandas as pd

Entrez.email = "dhruvk090@gmail.com"

search_term = "Mus musculus[Organism]"
filters = "Primary submission[Filter] AND Transcriptome or Gene expression[Filter]"

handle = Entrez.esearch(
    db="bioproject", term=search_term + " AND " + filters, retmax=100000
)
rec_list = Entrez.read(handle)
handle.close()

id_list = rec_list["IdList"]
print(len(id_list))
handle = Entrez.efetch(db="bioproject", id=id_list, rettype="xml", retmode="xml")
xml_data = handle.read().decode("utf-8")
handle.close()
root = ET.fromstring(xml_data)

data = []
samples_data = []
sample_ids = []
i = 0


def read_text_from_xml(file, link):
    value = file.find(link)
    if value is not None:
        value = value.text
    else:
        value = "NA"
    return value


def get_value_from_xml(file, link, getter):
    value = file.find(link)
    if value is not None:
        value = value.get(getter)
    else:
        value = "NA"
    return value


for doc in root.findall("DocumentSummary"):
    _projectID = get_value_from_xml(doc, "Project/ProjectID/ArchiveID", "accession")
    _id = get_value_from_xml(doc, "Project/ProjectID/ArchiveID", "id")
    name = read_text_from_xml(doc, "Project/ProjectDescr/Name")
    title = read_text_from_xml(doc, "Project/ProjectDescr/Title")
    description = read_text_from_xml(doc, "Project/ProjectDescr/Description")

    data.append(
        {
            "ID": _id,
            "Project ID": _projectID,
            "Project Name": name,
            "Title": title,
            "Description": description,
        }
    )

    handle = Entrez.elink(dbfrom="bioproject", id=_id, db="biosample", retmax=100000)
    record = Entrez.read(handle)
    handle.close()
    if not record or not record[0]["LinkSetDb"]:
        print("No BioSamples found")
    else:
        sample_ids.extend([link["Id"] for link in record[0]["LinkSetDb"][0]["Link"]])

projectDF = pd.DataFrame(data)
projectDF.to_csv("bioProjects.csv")
print("Projects Saved")

for i in range(0, len(sample_ids), 100):
    handle = Entrez.efetch(
        db="biosample", id=sample_ids[i : i + 100], rettype="xml", retmode="xml"
    )
    xml_data = handle.read().decode("utf-8")
    handle.close()
    sample_root = ET.fromstring(xml_data)
    for sample in sample_root.findall("BioSample"):
        sampleID = sample.get("accession")
        title = read_text_from_xml(sample, "Description/Title")
        BioProjectID = read_text_from_xml(sample, 'Links/Link[@target="bioproject"]')
        attributes = {}
        for attribute in sample.findall("Attributes/Attribute"):
            attribute_name = attribute.get("display_name")
            attribute_value = attribute.text
            attributes[attribute_name] = attribute_value
        row = {"sampleID": sampleID, "Title": title, "BioProjectID": BioProjectID}
        for attribute_name, attribute_value in attributes.items():
            row[attribute_name] = attribute_value
        samples_data.append(row)

samplesDF = pd.DataFrame(samples_data)
samplesDF.to_csv("bioSamples.csv")
