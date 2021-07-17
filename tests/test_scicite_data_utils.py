from utils.scicite_data_utils import *
import os
import pytest


@pytest.mark.preprocess
def test_process_apa_citation():
    test_cases = [
        {
            'origin':  "With regard to family fit, there is evidence that a minority of deaf couples would prefer to have deaf children and would consider using prenatal diagnosis to identify and terminate a hearing fetus (#REF).",
            'preprocessed':  "With regard to family fit, there is evidence that a minority of deaf couples would prefer to have deaf children and would consider using prenatal diagnosis to identify and terminate a hearing fetus #REF.",
        },
        {
            'origin': "In this study, all coated NiTi wires showed more deflection with the unloading force than the loading force, which is consistent with previously reported\nresults (Lim et al., \u201994; Krishnan and Kumar, 2004; Elayyan et al., 2008, 2010; Alavi and Hosseini, 2012; #REF).",
            'preprocessed': "In this study, all coated NiTi wires showed more deflection with the unloading force than the loading force, which is consistent with previously reported\nresults #REF.",
        },
        {
            'origin': "In the study by #REF, spikes were sampled from the field at the point of physiological\nrobinson et al.: genomic regions influencing root traits in barley 11 of 13\nmaturity, dried, grain threshed by hand, and stored at \u221220C to preserve grain dormancy before germination testing.",
            'preprocessed': "In the study by #REF, spikes were sampled from the field at the point of physiological\nrobinson et al.: genomic regions influencing root traits in barley 11 of 13\nmaturity, dried, grain threshed by hand, and stored at \u221220C to preserve grain dormancy before germination testing.",
        },
        {
            'origin': "In this study, all coated NiTi wires (Lim et al., \u201994; Krishnan and Kumar, 2004; Elayyan et al., 2008, 2010; Alavi and Hosseini, 2012) showed more deflection with the unloading force than the loading force, which is consistent with previously reported\nresults (Lim et al., \u201994; Krishnan and Kumar, 2004; Elayyan et al., 2008, 2010; Alavi and Hosseini, 2012; #REF).",
            'preprocessed': "In this study, all coated NiTi wires #OTHERREF showed more deflection with the unloading force than the loading force, which is consistent with previously reported\nresults #REF.",
        },
        {
            'origin': "Most studies focused on the relation between brain activation and motor function (#REF; Staudt et al. 2002) or the effects of intervention on brain activation (Golomb et al. 2010; Sutcliffe et al. 2009; Walther et al. 2009; You et al. 2005; Cope et al. 2010).",
            'preprocessed': "Most studies focused on the relation between brain activation and motor function #REF or the effects of intervention on brain activation #OTHERREF."
        }
    ]

    for test_case in test_cases:
        assert process_apa_citation(
            test_case['origin']) == test_case['preprocessed']


@pytest.mark.preprocess
def test_process_ieee_citation():
    test_cases = [
        {
            'origin':  "The language models were characterN-grams trained using texts from the Corpus of Spontaneous Japanese (CSJ) #REF, which contains 12.",
            'preprocessed':  "The language models were characterN-grams trained using texts from the Corpus of Spontaneous Japanese (CSJ) #REF, which contains 12.",
        },
        {
            'origin': "Other forms of elevated CK signaling, such as CK application or overexpression of IPT [39, 40] or KNOTTED genes [41], were insufficient to promote tubers in tomato, probably due to inappropriate local hormonal balances or to changing ratios among CK metabolites #REF.",
            'preprocessed': "Other forms of elevated CK signaling, such as CK application or overexpression of IPT #OTHERREF or KNOTTED genes #OTHERREF, were insufficient to promote tubers in tomato, probably due to inappropriate local hormonal balances or to changing ratios among CK metabolites #REF.",
        },
        {
            'origin':  "The language models [5-7] were characterN-grams [8-12, 17] trained using texts from the Corpus of Spontaneous Japanese (CSJ) #REF, which contains 12.",
            'preprocessed':  "The language models #OTHERREF were characterN-grams #OTHERREF trained using texts from the Corpus of Spontaneous Japanese (CSJ) #REF, which contains 12.",
        },
        {
            'origin':  "The language models [5-7] were characterN-grams [8-12,17] trained using texts from the Corpus of Spontaneous Japanese (CSJ) #REF, which contains 12.",
            'preprocessed':  "The language models #OTHERREF were characterN-grams #OTHERREF trained using texts from the Corpus of Spontaneous Japanese (CSJ) #REF, which contains 12.",
        },
        {
            'origin':  "The language models [5-7] were characterN-grams [8-12,#REF] trained using texts from the Corpus of Spontaneous Japanese (CSJ) [23], which contains 12.",
            'preprocessed':  "The language models #OTHERREF were characterN-grams #REF trained using texts from the Corpus of Spontaneous Japanese (CSJ) #OTHERREF, which contains 12.",
        },
    ]

    for test_case in test_cases:
        assert process_ieee_citation(
            test_case['origin']) == test_case['preprocessed']


@pytest.mark.preprocess
def test_get_citation_marker_from_span():
    test_cases = [
        {
            'origin':  "The language models were characterN-grams trained using texts from the Corpus of Spontaneous Japanese (CSJ) [23], which contains 12.",
            'citation_marker':  "[23]",
            'citation_marker_span': (108, 112)
        },
        {
            'origin':  "With regard to family fit, there is evidence that a minority of deaf couples would prefer to have deaf children and would consider using prenatal diagnosis to identify and terminate a hearing fetus (Middleton et al., 2001).",
            'citation_marker':  "Middleton et al., 2001",
            'citation_marker_span': (199, 221)
        },
        {
            'origin': "Other forms of elevated CK signaling, such as CK application or overexpression of IPT [39, 40] or KNOTTED genes [41], were insufficient to promote tubers in tomato, probably due to inappropriate local hormonal balances or to changing ratios among CK metabolites [42].",
            'citation_marker': "[42]",
            'citation_marker_span': (262, 266)
        },
        {
            'origin': "Most studies focused on the relation between brain activation and motor function (Guzzetta et al. 2007; Staudt et al. 2002) or the effects of intervention on brain activation (Golomb et al. 2010; Sutcliffe et al. 2009; Walther et al. 2009; You et al. 2005; Cope et al. 2010).",
            'citation_marker': "Guzzetta et al. 2007",
            'citation_marker_span': (82, 102)
        }
    ]

    for test_case in test_cases:
        assert get_citation_marker_from_span(
            test_case['origin'], test_case['citation_marker_span']) == test_case['citation_marker']


@pytest.mark.preprocess
def test_process_reference_marker():
    test_cases = [
        {
            'origin':  "The language models were characterN-grams trained using texts from the Corpus of Spontaneous Japanese (CSJ) [23], which contains 12.",
            'preprocessed':  "The language models were characterN-grams trained using texts from the Corpus of Spontaneous Japanese (CSJ) #REF, which contains 12.",
            'citation_marker': "[23]"
        },
        {
            'origin':  "With regard to family fit, there is evidence that a minority of deaf couples would prefer to have deaf children and would consider using prenatal diagnosis to identify and terminate a hearing fetus (Middleton et al., 2001).",
            'preprocessed':  "With regard to family fit, there is evidence that a minority of deaf couples would prefer to have deaf children and would consider using prenatal diagnosis to identify and terminate a hearing fetus #REF.",
            'citation_marker': "Middleton et al., 2001"
        },
        {
            'origin': "Other forms of elevated CK signaling, such as CK application or overexpression of IPT [39, 40] or KNOTTED genes [41], were insufficient to promote tubers in tomato, probably due to inappropriate local hormonal balances or to changing ratios among CK metabolites [42].",
            'preprocessed': "Other forms of elevated CK signaling, such as CK application or overexpression of IPT #OTHERREF or KNOTTED genes #OTHERREF, were insufficient to promote tubers in tomato, probably due to inappropriate local hormonal balances or to changing ratios among CK metabolites #REF.",
            'citation_marker': "[42]"
        },
        {
            'origin': "The Jacobi\u2013Davidson methods can be viewed as instances of Newton\u2019s method with subspace acceleration for the solution of the given eigenproblem, restricted to vectors located on the unit ball [40].",
            'preprocessed': "The Jacobi\u2013Davidson methods can be viewed as instances of Newton\u2019s method with subspace acceleration for the solution of the given eigenproblem, restricted to vectors located on the unit ball #REF.",
            'citation_marker': "[40]"
        }
    ]

    for test_case in test_cases:
        assert process_reference_marker(
            test_case['origin'], test_case['citation_marker']) == test_case['preprocessed']


@pytest.mark.preprocess
def test_get_citing_paper_input():
    test_cases = [
        {
            'data_pair': {
                'citingTitle': "Genomic Regions Influencing Seminal Root Traits in Barley",
                'citingAbstract': "Water availability is a major limiting factor for crop production, making drought adaptation and its many component traits a desirable attribute of plant cultivars. Previous studies in cereal crops indicate that root traits expressed at early plant developmental stages, such as seminal root angle and root number, are associated with water extraction at different depths. Here, we conducted the first study to map seminal root traits in barley (Hordeum vulgare L.). Using a recently developed high\u2010throughput phenotyping method, a panel of 30 barley genotypes and a doubled\u2010haploid (DH) population (ND24260 \u00d7 \u2018Flagship\u2019) comprising 330 lines genotyped with diversity array technology (DArT) markers were evaluated for seminal root angle (deviation from vertical) and root number under controlled environmental conditions. A high degree of phenotypic variation was observed in the panel of 30 genotypes: 13.5 to 82.2 and 3.6 to 6.9\u00b0 for root angle and root number, respectively. A similar range was observed in the DH population: 16.4 to 70.5 and 3.6 to 6.5\u00b0 for root angle and number, respectively. Seven quantitative trait loci (QTL) for seminal root traits (root angle, two QTL; root number, five QTL) were detected in the DH population. A major QTL influencing both root angle and root number (RAQ2/RNQ4) was positioned on chromosome 5HL. Across\u2010species analysis identified 10 common genes underlying root trait QTL in barley, wheat (Triticum aestivum L.), and sorghum [Sorghum bicolor (L.) Moench]. Here, we provide insight into seminal root phenotypes and provide a first look at the genetics controlling these traits in barley.",
                'intent': "background"
            },
            'citing_input_mode': "title",
            "target": "Genomic Regions Influencing Seminal Root Traits in Barley",
        },
        {
            'data_pair': {
                'citingTitle': "Genomic Regions Influencing Seminal Root Traits in Barley",
                'citingAbstract': "Water availability is a major limiting factor for crop production, making drought adaptation and its many component traits a desirable attribute of plant cultivars. Previous studies in cereal crops indicate that root traits expressed at early plant developmental stages, such as seminal root angle and root number, are associated with water extraction at different depths. Here, we conducted the first study to map seminal root traits in barley (Hordeum vulgare L.). Using a recently developed high\u2010throughput phenotyping method, a panel of 30 barley genotypes and a doubled\u2010haploid (DH) population (ND24260 \u00d7 \u2018Flagship\u2019) comprising 330 lines genotyped with diversity array technology (DArT) markers were evaluated for seminal root angle (deviation from vertical) and root number under controlled environmental conditions. A high degree of phenotypic variation was observed in the panel of 30 genotypes: 13.5 to 82.2 and 3.6 to 6.9\u00b0 for root angle and root number, respectively. A similar range was observed in the DH population: 16.4 to 70.5 and 3.6 to 6.5\u00b0 for root angle and number, respectively. Seven quantitative trait loci (QTL) for seminal root traits (root angle, two QTL; root number, five QTL) were detected in the DH population. A major QTL influencing both root angle and root number (RAQ2/RNQ4) was positioned on chromosome 5HL. Across\u2010species analysis identified 10 common genes underlying root trait QTL in barley, wheat (Triticum aestivum L.), and sorghum [Sorghum bicolor (L.) Moench]. Here, we provide insight into seminal root phenotypes and provide a first look at the genetics controlling these traits in barley.",
                'intent': "background"
            },
            'citing_input_mode': "abstract",
            "target": "Water availability is a major limiting factor for crop production, making drought adaptation and its many component traits a desirable attribute of plant cultivars. Previous studies in cereal crops indicate that root traits expressed at early plant developmental stages, such as seminal root angle and root number, are associated with water extraction at different depths. Here, we conducted the first study to map seminal root traits in barley (Hordeum vulgare L.). Using a recently developed high\u2010throughput phenotyping method, a panel of 30 barley genotypes and a doubled\u2010haploid (DH) population (ND24260 \u00d7 \u2018Flagship\u2019) comprising 330 lines genotyped with diversity array technology (DArT) markers were evaluated for seminal root angle (deviation from vertical) and root number under controlled environmental conditions. A high degree of phenotypic variation was observed in the panel of 30 genotypes: 13.5 to 82.2 and 3.6 to 6.9\u00b0 for root angle and root number, respectively. A similar range was observed in the DH population: 16.4 to 70.5 and 3.6 to 6.5\u00b0 for root angle and number, respectively. Seven quantitative trait loci (QTL) for seminal root traits (root angle, two QTL; root number, five QTL) were detected in the DH population. A major QTL influencing both root angle and root number (RAQ2/RNQ4) was positioned on chromosome 5HL. Across\u2010species analysis identified 10 common genes underlying root trait QTL in barley, wheat (Triticum aestivum L.), and sorghum [Sorghum bicolor (L.) Moench]. Here, we provide insight into seminal root phenotypes and provide a first look at the genetics controlling these traits in barley.",
        },
        {
            'data_pair': {
                'citingTitle': "Genomic Regions Influencing Seminal Root Traits in Barley",
                'citingAbstract': "Water availability is a major limiting factor for crop production, making drought adaptation and its many component traits a desirable attribute of plant cultivars. Previous studies in cereal crops indicate that root traits expressed at early plant developmental stages, such as seminal root angle and root number, are associated with water extraction at different depths. Here, we conducted the first study to map seminal root traits in barley (Hordeum vulgare L.). Using a recently developed high\u2010throughput phenotyping method, a panel of 30 barley genotypes and a doubled\u2010haploid (DH) population (ND24260 \u00d7 \u2018Flagship\u2019) comprising 330 lines genotyped with diversity array technology (DArT) markers were evaluated for seminal root angle (deviation from vertical) and root number under controlled environmental conditions. A high degree of phenotypic variation was observed in the panel of 30 genotypes: 13.5 to 82.2 and 3.6 to 6.9\u00b0 for root angle and root number, respectively. A similar range was observed in the DH population: 16.4 to 70.5 and 3.6 to 6.5\u00b0 for root angle and number, respectively. Seven quantitative trait loci (QTL) for seminal root traits (root angle, two QTL; root number, five QTL) were detected in the DH population. A major QTL influencing both root angle and root number (RAQ2/RNQ4) was positioned on chromosome 5HL. Across\u2010species analysis identified 10 common genes underlying root trait QTL in barley, wheat (Triticum aestivum L.), and sorghum [Sorghum bicolor (L.) Moench]. Here, we provide insight into seminal root phenotypes and provide a first look at the genetics controlling these traits in barley.",
                'intent': "background"
            },
            'citing_input_mode': "hybrid",
            "target": "Genomic Regions Influencing Seminal Root Traits in Barley",
        },
        {
            'data_pair': {
                'citingTitle': "Genomic Regions Influencing Seminal Root Traits in Barley",
                'citingAbstract': "Water availability is a major limiting factor for crop production, making drought adaptation and its many component traits a desirable attribute of plant cultivars. Previous studies in cereal crops indicate that root traits expressed at early plant developmental stages, such as seminal root angle and root number, are associated with water extraction at different depths. Here, we conducted the first study to map seminal root traits in barley (Hordeum vulgare L.). Using a recently developed high\u2010throughput phenotyping method, a panel of 30 barley genotypes and a doubled\u2010haploid (DH) population (ND24260 \u00d7 \u2018Flagship\u2019) comprising 330 lines genotyped with diversity array technology (DArT) markers were evaluated for seminal root angle (deviation from vertical) and root number under controlled environmental conditions. A high degree of phenotypic variation was observed in the panel of 30 genotypes: 13.5 to 82.2 and 3.6 to 6.9\u00b0 for root angle and root number, respectively. A similar range was observed in the DH population: 16.4 to 70.5 and 3.6 to 6.5\u00b0 for root angle and number, respectively. Seven quantitative trait loci (QTL) for seminal root traits (root angle, two QTL; root number, five QTL) were detected in the DH population. A major QTL influencing both root angle and root number (RAQ2/RNQ4) was positioned on chromosome 5HL. Across\u2010species analysis identified 10 common genes underlying root trait QTL in barley, wheat (Triticum aestivum L.), and sorghum [Sorghum bicolor (L.) Moench]. Here, we provide insight into seminal root phenotypes and provide a first look at the genetics controlling these traits in barley.",
                'intent': "result"
            },
            'citing_input_mode': "hybrid",
            "target": "Water availability is a major limiting factor for crop production, making drought adaptation and its many component traits a desirable attribute of plant cultivars. Previous studies in cereal crops indicate that root traits expressed at early plant developmental stages, such as seminal root angle and root number, are associated with water extraction at different depths. Here, we conducted the first study to map seminal root traits in barley (Hordeum vulgare L.). Using a recently developed high\u2010throughput phenotyping method, a panel of 30 barley genotypes and a doubled\u2010haploid (DH) population (ND24260 \u00d7 \u2018Flagship\u2019) comprising 330 lines genotyped with diversity array technology (DArT) markers were evaluated for seminal root angle (deviation from vertical) and root number under controlled environmental conditions. A high degree of phenotypic variation was observed in the panel of 30 genotypes: 13.5 to 82.2 and 3.6 to 6.9\u00b0 for root angle and root number, respectively. A similar range was observed in the DH population: 16.4 to 70.5 and 3.6 to 6.5\u00b0 for root angle and number, respectively. Seven quantitative trait loci (QTL) for seminal root traits (root angle, two QTL; root number, five QTL) were detected in the DH population. A major QTL influencing both root angle and root number (RAQ2/RNQ4) was positioned on chromosome 5HL. Across\u2010species analysis identified 10 common genes underlying root trait QTL in barley, wheat (Triticum aestivum L.), and sorghum [Sorghum bicolor (L.) Moench]. Here, we provide insight into seminal root phenotypes and provide a first look at the genetics controlling these traits in barley.",
        },
        {
            'data_pair': {
                'citingTitle': "Genomic Regions Influencing Seminal Root Traits in Barley",
                'citingAbstract': "Water availability is a major limiting factor for crop production, making drought adaptation and its many component traits a desirable attribute of plant cultivars. Previous studies in cereal crops indicate that root traits expressed at early plant developmental stages, such as seminal root angle and root number, are associated with water extraction at different depths. Here, we conducted the first study to map seminal root traits in barley (Hordeum vulgare L.). Using a recently developed high\u2010throughput phenotyping method, a panel of 30 barley genotypes and a doubled\u2010haploid (DH) population (ND24260 \u00d7 \u2018Flagship\u2019) comprising 330 lines genotyped with diversity array technology (DArT) markers were evaluated for seminal root angle (deviation from vertical) and root number under controlled environmental conditions. A high degree of phenotypic variation was observed in the panel of 30 genotypes: 13.5 to 82.2 and 3.6 to 6.9\u00b0 for root angle and root number, respectively. A similar range was observed in the DH population: 16.4 to 70.5 and 3.6 to 6.5\u00b0 for root angle and number, respectively. Seven quantitative trait loci (QTL) for seminal root traits (root angle, two QTL; root number, five QTL) were detected in the DH population. A major QTL influencing both root angle and root number (RAQ2/RNQ4) was positioned on chromosome 5HL. Across\u2010species analysis identified 10 common genes underlying root trait QTL in barley, wheat (Triticum aestivum L.), and sorghum [Sorghum bicolor (L.) Moench]. Here, we provide insight into seminal root phenotypes and provide a first look at the genetics controlling these traits in barley.",
                'intent': "method"
            },
            'citing_input_mode': "hybrid",
            "target": "Genomic Regions Influencing Seminal Root Traits in Barley",
        },
        {
            'data_pair': {
                'citingTitle': "Genomic Regions Influencing Seminal Root Traits in Barley",
                'citingAbstract': "Water availability is a major limiting factor for crop production, making drought adaptation and its many component traits a desirable attribute of plant cultivars. Previous studies in cereal crops indicate that root traits expressed at early plant developmental stages, such as seminal root angle and root number, are associated with water extraction at different depths. Here, we conducted the first study to map seminal root traits in barley (Hordeum vulgare L.). Using a recently developed high\u2010throughput phenotyping method, a panel of 30 barley genotypes and a doubled\u2010haploid (DH) population (ND24260 \u00d7 \u2018Flagship\u2019) comprising 330 lines genotyped with diversity array technology (DArT) markers were evaluated for seminal root angle (deviation from vertical) and root number under controlled environmental conditions. A high degree of phenotypic variation was observed in the panel of 30 genotypes: 13.5 to 82.2 and 3.6 to 6.9\u00b0 for root angle and root number, respectively. A similar range was observed in the DH population: 16.4 to 70.5 and 3.6 to 6.5\u00b0 for root angle and number, respectively. Seven quantitative trait loci (QTL) for seminal root traits (root angle, two QTL; root number, five QTL) were detected in the DH population. A major QTL influencing both root angle and root number (RAQ2/RNQ4) was positioned on chromosome 5HL. Across\u2010species analysis identified 10 common genes underlying root trait QTL in barley, wheat (Triticum aestivum L.), and sorghum [Sorghum bicolor (L.) Moench]. Here, we provide insight into seminal root phenotypes and provide a first look at the genetics controlling these traits in barley.",
                'intent': "background"
            },
            'citing_input_mode': "none",
            "target": "",
        },
    ]
    for test_case in test_cases:
        data_pair = test_case['data_pair']
        assert test_case['target'] == get_citing_paper_input(
            data_pair, test_case['citing_input_mode'])
