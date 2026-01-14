import { GoogleGenerativeAI } from '@google/generative-ai';

// Lazy-initialize Gemini API (called after dotenv loads)
let genAI = null;

function getGenAI() {
    if (!genAI && process.env.GEMINI_API_KEY) {
        console.log('🤖 Initializing Gemini AI with API key...');
        genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    }
    return genAI;
}

/**
 * Generate comprehensive pathology report using Gemini
 */
export async function generateFullReport(sample, mlResults) {
    const ai = getGenAI();
    if (!ai) {
        console.warn('⚠️ GEMINI_API_KEY not set - using fallback');
        return { success: false, error: 'Gemini API key not configured' };
    }

    try {
        console.log('🤖 Calling Gemini API for full report...');
        const model = ai.getGenerativeModel({ model: 'gemini-2.5-flash' });

        const prompt = `You are an expert pathologist AI assistant. Generate a professional pathology report based on the following data.

SAMPLE INFORMATION:
- Sample ID: ${sample.sampleId || sample._id}
- Sample Type: ${sample.sampleType || 'Tissue Biopsy'}
- Collection Date: ${sample.sampleInfo?.collectionDate || 'Not specified'}

PATIENT INFORMATION:
- Name: ${sample.patientInfo?.name || 'Not provided'}
- Age: ${sample.patientInfo?.age || 'Unknown'}
- Gender: ${sample.patientInfo?.gender || 'Unknown'}

AI ANALYSIS RESULTS:
- Total Images Analyzed: ${mlResults.totalImages || 0}
- Detection Result: ${mlResults.isPositive ? 'POSITIVE (Abnormal)' : 'NEGATIVE (Normal)'}
- Tumor/Detection Probability: ${mlResults.tumorProbability?.toFixed(1) || 0}%
- AI Confidence: ${mlResults.confidence?.toFixed(1) || 0}%
- Risk Level: ${mlResults.riskLevel || 'Unknown'}

Generate a JSON response with the following structure:
{
  "clinicalSummary": "2-3 sentence professional summary of findings",
  "interpretation": "Detailed interpretation of the AI analysis results (3-4 sentences)",
  "recommendations": ["Array of 2-3 clinical recommendations"],
  "morphologicalFindings": "Description of tissue/cell morphology based on results",
  "conclusion": "Final diagnostic conclusion (1-2 sentences)"
}

Be professional, use appropriate medical terminology, and always include disclaimers about AI-assisted analysis requiring pathologist confirmation.`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text().trim();

        // Parse JSON from response
        let reportContent;
        try {
            // Extract JSON from potential markdown code blocks
            const jsonMatch = text.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                reportContent = JSON.parse(jsonMatch[0]);
            } else {
                throw new Error('No JSON found in response');
            }
        } catch (parseError) {
            console.error('Failed to parse Gemini response:', parseError);
            reportContent = {
                clinicalSummary: text.substring(0, 300),
                interpretation: 'AI-generated interpretation available.',
                recommendations: ['Consult with pathologist for confirmation'],
                morphologicalFindings: 'See detailed analysis above.',
                conclusion: 'AI analysis complete. Pathologist review recommended.'
            };
        }

        return {
            success: true,
            generatedBy: 'Gemini AI',
            timestamp: new Date().toISOString(),
            content: reportContent
        };

    } catch (error) {
        console.error('Gemini full report error:', error);
        return {
            success: false,
            error: error.message,
            generatedBy: 'Error'
        };
    }
}

/**
 * Generate clinical summary from ML analysis results using Gemini
 */
export async function generateClinicalSummary(mlResults, patientInfo = {}) {
    const ai = getGenAI();
    if (!ai) {
        console.warn('⚠️ GEMINI_API_KEY not set, using fallback summary');
        return generateFallbackSummary(mlResults);
    }

    try {
        console.log('🤖 Calling Gemini API for clinical summary...');
        const model = ai.getGenerativeModel({ model: 'gemini-2.5-flash' });

        const prompt = `You are a medical AI assistant generating clinical summaries for pathology reports. 

AI ANALYSIS DATA:
- Sample Type: ${mlResults.sampleType || 'Tissue Biopsy'}
- Images Analyzed: ${mlResults.totalImages || 0}
- Malignant Detections: ${mlResults.malignantDetections || 0}
- Detection Probability: ${((mlResults.tumorProbability || 0)).toFixed(1)}%
- AI Confidence: ${((mlResults.averageConfidence || 0) * 100).toFixed(1)}%
- Risk Assessment: ${mlResults.overallRisk || 'Unknown'}

PATIENT CONTEXT:
- Age: ${patientInfo.age || 'Unknown'}
- Gender: ${patientInfo.gender || 'Unknown'}

Generate a professional 2-3 sentence clinical summary suitable for a pathology report. Use medical terminology appropriate for physicians. Be factual, use phrases like "findings suggest" or "consistent with". Include confidence level and recommendations if abnormal.`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const summary = response.text().trim();

        return {
            success: true,
            summary: summary,
            generatedBy: 'Gemini AI',
            timestamp: new Date().toISOString()
        };
    } catch (error) {
        console.error('Gemini API error:', error);
        return generateFallbackSummary(mlResults);
    }
}

/**
 * Generate follow-up recommendations
 */
export async function generateRecommendations(mlResults) {
    const ai = getGenAI();
    if (!ai) {
        return generateFallbackRecommendations(mlResults);
    }

    try {
        console.log('🤖 Calling Gemini API for recommendations...');
        const model = ai.getGenerativeModel({ model: 'gemini-2.5-flash' });

        const prompt = `Based on these pathology AI analysis results, provide 3 brief clinical recommendations:

- Detection Result: ${mlResults.isPositive ? 'POSITIVE (Abnormal)' : 'NEGATIVE (Normal)'}
- Confidence: ${mlResults.confidence?.toFixed(1) || 0}%
- Risk Level: ${mlResults.riskLevel || 'Unknown'}

Return ONLY a JSON array of 3 recommendation strings, nothing else. Example format:
["Recommendation 1", "Recommendation 2", "Recommendation 3"]`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text().trim();

        let recommendations;
        try {
            recommendations = JSON.parse(text);
        } catch {
            recommendations = [
                'Consult with pathologist for confirmation',
                'Consider clinical correlation',
                'Follow standard care protocols'
            ];
        }

        return {
            success: true,
            recommendations,
            generatedBy: 'Gemini AI'
        };
    } catch (error) {
        console.error('Gemini recommendations error:', error);
        return generateFallbackRecommendations(mlResults);
    }
}

// Fallback functions
function generateFallbackSummary(mlResults) {
    const isPositive = mlResults.isPositive || (mlResults.malignantDetections || 0) > 0;
    // Confidence is already a percentage from routes, don't multiply again
    const rawConfidence = mlResults.averageConfidence || mlResults.confidence || 0;
    const confidence = (rawConfidence > 1 ? rawConfidence : rawConfidence * 100).toFixed(1);
    const riskLevel = mlResults.overallRisk || mlResults.riskLevel || 'Unknown';

    const summary = isPositive
        ? `AI-assisted analysis detected potential abnormalities in the submitted sample with ${confidence}% confidence. Risk assessment: ${riskLevel}. Further clinical evaluation and histopathological confirmation is recommended.`
        : `AI-assisted analysis of the submitted sample shows no significant abnormalities. Detection confidence: ${confidence}%. Risk level: ${riskLevel}. Normal tissue architecture and cellular morphology observed.`;

    return {
        success: true,
        summary,
        generatedBy: 'Fallback (No API Key)',
        timestamp: new Date().toISOString()
    };
}

function generateFallbackRecommendations(mlResults) {
    const isPositive = mlResults.isPositive || (mlResults.malignantDetections || 0) > 0;

    const recommendations = isPositive
        ? [
            'Recommend confirmatory biopsy with histopathological examination',
            'Consider additional imaging studies if clinically indicated',
            'Schedule follow-up consultation with specialist'
        ]
        : [
            'No immediate follow-up required based on current findings',
            'Continue routine screening as per clinical guidelines',
            'Consult with ordering physician for clinical correlation'
        ];

    return {
        success: true,
        recommendations,
        generatedBy: 'Fallback'
    };
}
