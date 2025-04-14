import os
import unittest
from unittest.mock import MagicMock, patch

from app.processors.l0.document_analyzer import DocumentInsightGenerator, DocumentSummaryGenerator
from app.processors.l0.document_insight_processor import DocumentInsightProcessor
from app.processors.l0.models import DocumentInsight, DocumentSummary

# Sample content for testing
SAMPLE_CONTENT = """
# Climate Change: Global Impacts and Mitigation Strategies

Climate change represents one of the most significant challenges facing humanity in the 21st century. 
This document examines the current understanding of climate change, its observed and projected impacts, 
and outlines key strategies for mitigation and adaptation.

## Current State of Climate Science

The Intergovernmental Panel on Climate Change (IPCC) has established with high confidence that human 
activities, primarily the burning of fossil fuels and deforestation, have been the dominant cause 
of the observed warming since the mid-20th century. Global average temperatures have increased by 
approximately 1.1°C above pre-industrial levels.

Key observations include:
- Rising sea levels (3.6mm per year)
- Increased frequency of extreme weather events
- Shifting precipitation patterns
- Declining Arctic sea ice
- Ocean acidification

## Mitigation Strategies

Effective climate change mitigation requires a multi-faceted approach:

1. **Energy Transition**: Shifting from fossil fuels to renewable energy sources
2. **Energy Efficiency**: Improving efficiency in buildings, transportation, and industry
3. **Carbon Pricing**: Implementing carbon taxes or cap-and-trade systems
4. **Nature-Based Solutions**: Reforestation and ecosystem restoration
5. **Technological Innovation**: Developing carbon capture and storage technologies

## Adaptation Measures

Even with ambitious mitigation efforts, some climate impacts are now unavoidable. Adaptation measures include:

- Infrastructure improvements for climate resilience
- Water management systems for changing precipitation patterns
- Agricultural adaptations for changing growing conditions
- Public health measures for new disease vectors
- Managed retreat from areas of unavoidable impact

## International Cooperation

The Paris Agreement established a framework for global action, with countries submitting Nationally 
Determined Contributions (NDCs) to reduce emissions. However, current pledges remain insufficient 
to limit warming to well below 2°C above pre-industrial levels.

## Conclusion

Addressing climate change requires unprecedented global cooperation and transformation of energy, 
transportation, and food systems. While the challenges are substantial, the economic and technological 
opportunities presented by this transition can lead to more sustainable and equitable societies.
"""

class TestDocumentInsightProcessor(unittest.TestCase):
    """Test the two-stage document analysis process"""
    
    def setUp(self):
        """Set up test environment"""
        # Set environment variables for testing
        os.environ["OPENAI_API_KEY"] = "dummy-api-key"
        
        # Create mock insight and summary for testing
        self.mock_insight = DocumentInsight(
            title="Climate Change: Impacts and Strategies",
            insight="This document examines climate change science, impacts, and response strategies. "
                   "It covers the current state of climate science, noting a 1.1°C temperature increase "
                   "and various environmental impacts. The document outlines mitigation approaches "
                   "including energy transition, efficiency improvements, carbon pricing, nature-based "
                   "solutions, and technological innovation. Adaptation measures are discussed, "
                   "highlighting infrastructure improvements, water management, agricultural adaptations, "
                   "public health measures, and managed retreat from high-risk areas. The international "
                   "context includes the Paris Agreement framework, though current pledges remain "
                   "insufficient. The conclusion emphasizes the need for global cooperation and system "
                   "transformation, while noting potential economic and social benefits."
        )
        
        self.mock_summary = DocumentSummary(
            title="Climate Change: Impacts and Strategies",
            summary="This document provides an overview of climate change, its impacts, and response strategies. "
                   "Human activities have caused approximately 1.1°C of warming, resulting in sea level rise, "
                   "extreme weather, and other environmental changes. Mitigation requires energy transition, "
                   "efficiency, carbon pricing, and nature-based solutions, while adaptation is necessary for "
                   "unavoidable impacts. International cooperation through the Paris Agreement framework is "
                   "essential but currently insufficient.",
            keywords=["climate change", "global warming", "mitigation", "adaptation", 
                     "renewable energy", "Paris Agreement", "carbon pricing", 
                     "extreme weather", "sea level rise", "energy transition"]
        )
    
    @patch('app.processors.l0.document_analyzer.OpenAI')
    def test_insight_generator(self, mock_openai_class):
        """Test the DocumentInsightGenerator"""
        # Set up mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = (
            '{"title": "Climate Change: Impacts and Strategies", '
            '"insight": "This document examines climate change science, impacts, and response strategies..."}'
        )
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create generator and call method
        generator = DocumentInsightGenerator(api_key="dummy-key")
        result = generator.generate_insight(SAMPLE_CONTENT, "climate_report.md")
        
        # Verify result
        self.assertEqual(result.title, "Climate Change: Impacts and Strategies")
        self.assertTrue("document examines climate change" in result.insight)
        
        # Verify OpenAI was called correctly
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('app.processors.l0.document_analyzer.OpenAI')
    def test_summary_generator(self, mock_openai_class):
        """Test the DocumentSummaryGenerator"""
        # Set up mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = (
            '{"summary": "This document provides an overview of climate change...", '
            '"keywords": ["climate change", "global warming", "mitigation", "adaptation"]}'
        )
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create generator and call method
        generator = DocumentSummaryGenerator(api_key="dummy-key")
        result = generator.generate_summary(
            SAMPLE_CONTENT, 
            self.mock_insight,
            "climate_report.md"
        )
        
        # Verify result
        self.assertEqual(result.title, self.mock_insight.title)  # Should use same title as insight
        self.assertTrue("overview of climate change" in result.summary)
        self.assertIn("climate change", result.keywords)
        
        # Verify OpenAI was called correctly
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('app.processors.l0.document_insight_processor.DocumentInsightGenerator')
    @patch('app.processors.l0.document_insight_processor.DocumentSummaryGenerator')
    def test_document_insight_processor(self, mock_summary_gen, mock_insight_gen):
        """Test the DocumentInsightProcessor orchestrator"""
        # Set up mocks
        mock_insight_instance = MagicMock()
        mock_insight_instance.generate_insight.return_value = self.mock_insight
        mock_insight_gen.return_value = mock_insight_instance
        
        mock_summary_instance = MagicMock()
        mock_summary_instance.generate_summary.return_value = self.mock_summary
        mock_summary_gen.return_value = mock_summary_instance
        
        # Create processor and call method
        processor = DocumentInsightProcessor()
        result = processor.process_document(SAMPLE_CONTENT, "climate_report.md", "doc-123")
        
        # Verify result
        self.assertEqual(result["insight"], self.mock_insight)
        self.assertEqual(result["summary"], self.mock_summary)
        
        # Verify methods were called correctly
        mock_insight_instance.generate_insight.assert_called_once_with(SAMPLE_CONTENT, "climate_report.md")
        mock_summary_instance.generate_summary.assert_called_once_with(SAMPLE_CONTENT, self.mock_insight, "climate_report.md")


if __name__ == "__main__":
    unittest.main() 