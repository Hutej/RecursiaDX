import express from 'express';
import Report from '../models/Report.js';
import Sample from '../models/Sample.js';
import User from '../models/User.js';
import { catchAsync } from '../middleware/errorHandler.js';
import { verifyToken as auth } from '../middleware/auth.js';
import { reportValidations } from '../middleware/validation.js';
import { validationResult } from 'express-validator';

// Gemini AI Service for clinical summary generation
let geminiService = null;
try {
  geminiService = await import('../services/geminiService.js');
} catch (error) {
  console.log('ℹ️ Gemini service not available - using fallback summaries');
}

const router = express.Router();

// Generate comprehensive report for a sample
router.post('/generate/:sampleId',
  // auth, // Temporarily disabled for testing
  catchAsync(async (req, res) => {
    const { sampleId } = req.params;
    const { reportType = 'Preliminary', includeImages = true } = req.body;

    console.log('🔄 Generating report for sample:', sampleId);

    // Find the sample with ML analysis
    const sample = await Sample.findById(sampleId);
    if (!sample) {
      return res.status(404).json({
        success: false,
        message: 'Sample not found'
      });
    }

    // Generate report ID
    const reportCount = await Report.countDocuments();
    const reportId = `RPT-${new Date().getFullYear()}-${String(reportCount + 1).padStart(4, '0')}`;

    // Aggregate ML analysis from all images
    const mlAnalysisSummary = sample.images.reduce((summary, image) => {
      if (image.mlAnalysis) {
        summary.totalImages++;
        if (image.mlAnalysis.prediction === 'malignant') {
          summary.malignantDetections++;
        }
        summary.averageConfidence += (image.mlAnalysis.confidence || 0);

        // Map risk assessment to numeric score
        const riskScore = image.mlAnalysis.riskAssessment === 'high' ? 0.8 :
          image.mlAnalysis.riskAssessment === 'medium' ? 0.5 : 0.2;
        summary.maxRiskScore = Math.max(summary.maxRiskScore, riskScore);

        // Collect all detected features
        if (image.mlAnalysis.metadata?.detected_features) {
          summary.allFeatures.push(...image.mlAnalysis.metadata.detected_features);
        }
      }
      return summary;
    }, {
      totalImages: 0,
      malignantDetections: 0,
      averageConfidence: 0,
      maxRiskScore: 0,
      allFeatures: []
    });

    if (mlAnalysisSummary.totalImages > 0) {
      mlAnalysisSummary.averageConfidence /= mlAnalysisSummary.totalImages;
      mlAnalysisSummary.uniqueFeatures = [...new Set(mlAnalysisSummary.allFeatures)];
    }

    // Generate AI interpretation
    const aiInterpretation = generateAIInterpretation(sample, mlAnalysisSummary);

    // Create the report
    const reportData = {
      reportId,
      sampleId: sample._id,
      patientInfo: {
        patientId: sample.patientInfo?.patientId || 'Unknown',
        name: sample.patientInfo?.name || 'Unknown',
        age: sample.patientInfo?.age || 0,
        gender: sample.patientInfo?.gender || 'Unknown',
        contactNumber: sample.patientInfo?.contactNumber || '',
        email: sample.patientInfo?.email || ''
      },
      clinicalInfo: {
        orderingPhysician: sample.clinicalInfo?.orderingPhysician || '',
        clinicalHistory: sample.clinicalInfo?.clinicalHistory || '',
        provisionalDiagnosis: sample.clinicalInfo?.provisionalDiagnosis || sample.clinicalInfo?.clinicalDiagnosis || '',
        sampleSite: sample.specimenDetails?.site || '',
        sampleDate: sample.collectionInfo?.collectionDate || sample.createdAt
      },
      aiAnalysis: {
        totalImages: mlAnalysisSummary.totalImages,
        processedImages: mlAnalysisSummary.totalImages,
        malignantDetections: mlAnalysisSummary.malignantDetections,
        averageConfidence: mlAnalysisSummary.averageConfidence,
        maxRiskScore: mlAnalysisSummary.maxRiskScore,
        detectedFeatures: mlAnalysisSummary.uniqueFeatures,
        overallRisk: mlAnalysisSummary.maxRiskScore > 0.7 ? 'High' :
          mlAnalysisSummary.maxRiskScore > 0.4 ? 'Moderate' : 'Low',
        aiInterpretation
      },
      imageAnalysis: includeImages ? sample.images.map(img => ({
        filename: img.filename,
        originalName: img.originalName,
        mlAnalysis: img.mlAnalysis,
        staining: img.staining,
        magnification: img.magnification
      })) : [],
      reportType,
      analysisId: sample._id, // Use sample ID as analysis reference
      status: 'draft',
      generatedBy: req.user?.id || 'system', // Handle case when no user auth
      generatedAt: new Date(),
      version: '1.0'
    };

    const report = new Report(reportData);
    await report.save();

    // Update sample status to 'Under Review'
    sample.status = 'Under Review';
    sample.lastUpdated = new Date();

    // Initialize history array if it doesn't exist
    if (!sample.history) {
      sample.history = [];
    }

    sample.history.push({
      status: 'Under Review',
      timestamp: new Date(),
      updatedBy: req.user?.id || 'system',
      notes: 'Report generated automatically based on AI analysis'
    });
    await sample.save();

    res.json({
      success: true,
      message: 'Report generated successfully',
      data: {
        reportId: report.reportId,
        _id: report._id,
        status: report.status,
        aiAnalysis: report.aiAnalysis,
        generatedAt: report.generatedAt
      }
    });
  })
);

// Get report by ID
router.get('/:reportId',
  auth,
  catchAsync(async (req, res) => {
    const { reportId } = req.params;

    const report = await Report.findOne({
      $or: [
        { reportId },
        { _id: reportId }
      ]
    }).populate('generatedBy', 'name email role');

    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report not found'
      });
    }

    res.json({
      success: true,
      data: report
    });
  })
);

// Get all reports with pagination and filtering
router.get('/',
  auth,
  catchAsync(async (req, res) => {
    const {
      page = 1,
      limit = 10,
      status,
      reportType,
      dateFrom,
      dateTo,
      search
    } = req.query;

    const filter = {};

    if (status) filter.status = status;
    if (reportType) filter.reportType = reportType;

    if (dateFrom || dateTo) {
      filter.generatedAt = {};
      if (dateFrom) filter.generatedAt.$gte = new Date(dateFrom);
      if (dateTo) filter.generatedAt.$lte = new Date(dateTo);
    }

    if (search) {
      filter.$or = [
        { reportId: { $regex: search, $options: 'i' } },
        { 'patientInfo.name': { $regex: search, $options: 'i' } },
        { 'patientInfo.patientId': { $regex: search, $options: 'i' } }
      ];
    }

    const options = {
      page: parseInt(page),
      limit: parseInt(limit),
      sort: { generatedAt: -1 },
      populate: {
        path: 'generatedBy',
        select: 'name email role'
      }
    };

    const reports = await Report.paginate(filter, options);

    res.json({
      success: true,
      data: reports
    });
  })
);

// Update report status (draft -> review -> approved -> finalized)
router.patch('/:reportId/status',
  auth,
  catchAsync(async (req, res) => {
    const { reportId } = req.params;
    const { status, reviewNotes } = req.body;

    const validStatuses = ['draft', 'review', 'approved', 'finalized', 'rejected'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid status. Must be one of: ' + validStatuses.join(', ')
      });
    }

    const report = await Report.findOne({
      $or: [{ reportId }, { _id: reportId }]
    });

    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report not found'
      });
    }

    // Add to workflow history
    report.workflow.push({
      status,
      timestamp: new Date(),
      updatedBy: req.user.id,
      notes: reviewNotes
    });

    report.status = status;
    report.lastUpdated = new Date();

    if (status === 'finalized') {
      report.finalizedAt = new Date();
      report.finalizedBy = req.user.id;
    }

    await report.save();

    res.json({
      success: true,
      message: `Report ${status} successfully`,
      data: {
        reportId: report.reportId,
        status: report.status,
        lastUpdated: report.lastUpdated
      }
    });
  })
);

// Download report as PDF (placeholder for now)
router.get('/:reportId/download',
  auth,
  catchAsync(async (req, res) => {
    const { reportId } = req.params;
    const { format = 'pdf' } = req.query;

    const report = await Report.findOne({
      $or: [{ reportId }, { _id: reportId }]
    });

    if (!report) {
      return res.status(404).json({
        success: false,
        message: 'Report not found'
      });
    }

    // For now, return report data (in production, generate actual PDF)
    res.json({
      success: true,
      message: 'Report download ready',
      data: {
        reportId: report.reportId,
        format,
        downloadUrl: `/api/reports/${reportId}/pdf`, // Future implementation
        report
      }
    });
  })
);

// Generate AI clinical summary using Gemini
router.post('/generate-summary/:sampleId',
  // auth, // Temporarily disabled for testing
  catchAsync(async (req, res) => {
    const { sampleId } = req.params;

    console.log('🤖 Generating Gemini clinical summary for sample:', sampleId);

    const sample = await Sample.findById(sampleId);
    if (!sample) {
      return res.status(404).json({
        success: false,
        message: 'Sample not found'
      });
    }

    // Aggregate ML results
    const mlResults = {
      totalImages: sample.images?.length || 0,
      malignantDetections: 0,
      averageConfidence: 0,
      overallRisk: 'Unknown',
      tumorProbability: null,
      sampleType: sample.sampleType || 'Tissue Biopsy'
    };

    if (sample.images?.length > 0) {
      let totalConfidence = 0;
      let maxRiskScore = 0;

      sample.images.forEach(img => {
        if (img.mlAnalysis) {
          if (img.mlAnalysis.is_tumor || img.mlAnalysis.prediction === 'malignant') {
            mlResults.malignantDetections++;
          }
          totalConfidence += img.mlAnalysis.confidence || 0;

          const riskScore = img.mlAnalysis.riskAssessment === 'high' ? 0.8 :
            img.mlAnalysis.riskAssessment === 'medium' ? 0.5 : 0.2;
          maxRiskScore = Math.max(maxRiskScore, riskScore);

          if (img.mlAnalysis.metadata?.probabilities?.tumor) {
            mlResults.tumorProbability = img.mlAnalysis.metadata.probabilities.tumor;
          }
        }
      });

      mlResults.averageConfidence = totalConfidence / sample.images.length;
      mlResults.overallRisk = maxRiskScore > 0.7 ? 'High' : maxRiskScore > 0.4 ? 'Moderate' : 'Low';
    }

    // Generate summary using Gemini or fallback
    let summary;
    if (geminiService?.generateClinicalSummary) {
      summary = await geminiService.generateClinicalSummary(mlResults, sample.patientInfo || {});
    } else {
      // Fallback summary
      const hasMalignant = mlResults.malignantDetections > 0;
      const confidence = (mlResults.averageConfidence * 100).toFixed(1);

      summary = {
        success: true,
        summary: hasMalignant
          ? `AI analysis detected ${mlResults.malignantDetections} malignant region(s). Risk: ${mlResults.overallRisk}. Confidence: ${confidence}%. Further evaluation recommended.`
          : `AI analysis shows no significant abnormalities. Risk: ${mlResults.overallRisk}. Confidence: ${confidence}%. Normal findings.`,
        generatedBy: 'Fallback',
        timestamp: new Date().toISOString()
      };
    }

    res.json({
      success: true,
      data: {
        sampleId,
        mlResults,
        clinicalSummary: summary
      }
    });
  })
);

// Generate FULL AI Report using Gemini
router.post('/generate-full/:sampleId',
  catchAsync(async (req, res) => {
    const { sampleId } = req.params;

    console.log('🤖 Generating FULL Gemini report for sample:', sampleId);

    const sample = await Sample.findById(sampleId);
    if (!sample) {
      return res.status(404).json({
        success: false,
        message: 'Sample not found'
      });
    }

    // Build ML results object
    const mlResults = {
      totalImages: sample.images?.length || 0,
      isPositive: false,
      tumorProbability: 0,
      confidence: 0,
      riskLevel: 'Low',
      sampleType: sample.sampleType || 'Tissue Biopsy'
    };

    if (sample.images?.length > 0) {
      const firstImage = sample.images[0];
      if (firstImage?.mlAnalysis) {
        const analysis = firstImage.mlAnalysis;
        const tumorProb = (analysis.metadata?.probabilities?.tumor || 0) * 100;

        mlResults.tumorProbability = tumorProb;
        mlResults.confidence = (analysis.confidence || 0) * 100;
        mlResults.isPositive = tumorProb >= 54;
        mlResults.riskLevel = analysis.risk_level || analysis.riskAssessment || 'Low';
      }
    }

    // Generate full report using Gemini - SINGLE REQUEST to avoid rate limits
    let reportContent;
    if (geminiService?.generateFullReport) {
      console.log('🚀 Making SINGLE Gemini API request...');
      reportContent = await geminiService.generateFullReport(sample, mlResults);
    } else {
      // Fallback content
      reportContent = {
        success: true,
        generatedBy: 'Fallback',
        content: {
          clinicalSummary: mlResults.isPositive
            ? `AI analysis detected potential abnormalities with ${mlResults.confidence.toFixed(1)}% confidence.`
            : `AI analysis confirms normal findings with ${mlResults.confidence.toFixed(1)}% confidence.`,
          interpretation: 'The AI-powered digital pathology analysis has been completed. Results should be correlated with clinical findings.',
          recommendations: mlResults.isPositive
            ? ['Further histopathological examination recommended', 'Consider clinical correlation', 'Follow-up testing may be indicated']
            : ['No immediate follow-up required', 'Continue routine screening', 'Clinical correlation advised'],
          morphologicalFindings: mlResults.isPositive
            ? 'AI detected features suggestive of cellular abnormalities.'
            : 'Normal tissue architecture and cellular morphology observed.',
          conclusion: mlResults.isPositive
            ? 'Findings warrant pathologist review and possible additional testing.'
            : 'No significant abnormalities detected. Routine follow-up recommended.'
        }
      };
    }

    // Use the report content for summary and recommendations (no extra API calls)
    const summary = reportContent?.content?.clinicalSummary ? {
      success: true,
      summary: reportContent.content.clinicalSummary,
      generatedBy: reportContent.generatedBy || 'Gemini AI'
    } : null;

    const recommendations = reportContent?.content?.recommendations ? {
      success: true,
      recommendations: reportContent.content.recommendations,
      generatedBy: reportContent.generatedBy || 'Gemini AI'
    } : null;

    res.json({
      success: true,
      data: {
        sampleId,
        mlResults,
        report: reportContent,
        clinicalSummary: summary,
        recommendations: recommendations
      }
    });
  })
);

// Generate AI interpretation based on ML analysis
function generateAIInterpretation(sample, mlSummary) {
  const interpretations = [];

  if (mlSummary.totalImages === 0) {
    return "No images were available for AI analysis.";
  }

  // Overall assessment
  if (mlSummary.malignantDetections > 0) {
    interpretations.push(`AI analysis identified potential malignant features in ${mlSummary.malignantDetections} out of ${mlSummary.totalImages} images.`);
  } else {
    interpretations.push(`AI analysis did not identify malignant features in the examined ${mlSummary.totalImages} images.`);
  }

  // Confidence assessment
  const avgConfidence = (mlSummary.averageConfidence * 100).toFixed(1);
  interpretations.push(`Average prediction confidence: ${avgConfidence}%.`);

  // Risk assessment
  const riskScore = (mlSummary.maxRiskScore * 100).toFixed(0);
  if (mlSummary.maxRiskScore > 0.7) {
    interpretations.push(`High risk assessment (${riskScore}%) - recommend immediate pathologist review.`);
  } else if (mlSummary.maxRiskScore > 0.4) {
    interpretations.push(`Moderate risk assessment (${riskScore}%) - standard pathologist review recommended.`);
  } else {
    interpretations.push(`Low risk assessment (${riskScore}%) - routine pathologist review.`);
  }

  // Feature analysis
  if (mlSummary.uniqueFeatures.length > 0) {
    interpretations.push(`Detected features include: ${mlSummary.uniqueFeatures.join(', ')}.`);
  }

  interpretations.push("Note: AI analysis is a screening tool and should not replace expert pathologist interpretation.");

  return interpretations.join(' ');
}

// PDF Download endpoint
router.get('/download-pdf/:sampleId',
  catchAsync(async (req, res) => {
    const { sampleId } = req.params;

    console.log('📄 Generating PDF for sample:', sampleId);

    const sample = await Sample.findById(sampleId);
    if (!sample) {
      console.log('❌ Sample not found:', sampleId);
      return res.status(404).json({ success: false, message: 'Sample not found' });
    }

    console.log('✅ Sample found:', sample.sampleId || sampleId);

    // Import Puppeteer dynamically
    let puppeteer;
    try {
      puppeteer = await import('puppeteer');
      console.log('✅ Puppeteer imported');
    } catch (error) {
      console.error('❌ Failed to import Puppeteer:', error.message);
      return res.status(500).json({
        success: false,
        message: 'PDF generation unavailable - Puppeteer not installed',
        error: error.message
      });
    }

    try {
      // Build report data
      const patientInfo = sample.patientInfo || {};
      const firstImage = sample.images?.[0];
      const analysis = firstImage?.mlAnalysis || {};

      const tumorProb = ((analysis.metadata?.probabilities?.tumor || 0) * 100).toFixed(1);
      const confidence = ((analysis.confidence || 0) * 100).toFixed(1);
      const isPositive = parseFloat(tumorProb) >= 54;

      // Generate report HTML
      const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: 'Arial', sans-serif; 
      padding: 40px;
      color: #1a1a1a;
      line-height: 1.6;
    }
    .header {
      border-bottom: 4px solid #1e40af;
      padding-bottom: 20px;
      margin-bottom: 30px;
    }
    .header h1 {
      color: #1e40af;
      font-size: 28px;
      margin-bottom: 10px;
    }
    .header .info {
      display: flex;
      justify-content: space-between;
      margin-top: 15px;
      font-size: 12px;
      color: #666;
    }
    .section {
      margin-bottom: 25px;
      page-break-inside: avoid;
    }
    .section-title {
      background: #f3f4f6;
      padding: 10px 15px;
      font-size: 16px;
      font-weight: bold;
      color: #1e40af;
      margin-bottom: 15px;
      border-left: 4px solid #1e40af;
    }
    .field-group {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 15px;
      margin-bottom: 15px;
    }
    .field {
      padding: 10px;
      background: #f9fafb;
      border-radius: 4px;
    }
    .field-label {
      font-size: 11px;
      color: #6b7280;
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 5px;
    }
    .field-value {
      font-size: 14px;
      color: #1f2937;
      font-weight: 500;
    }
    .alert {
      padding: 15px;
      border-radius: 6px;
      margin: 15px 0;
    }
    .alert-positive {
      background: #fef2f2;
      border-left: 4px solid #dc2626;
      color: #991b1b;
    }
    .alert-negative {
      background: #f0fdf4;
      border-left: 4px solid #16a34a;
      color: #166534;
    }
    .alert-title {
      font-weight: bold;
      font-size: 16px;
      margin-bottom: 8px;
    }
    .footer {
      margin-top: 40px;
      padding-top: 20px;
      border-top: 2px solid #e5e7eb;
      font-size: 11px;
      color: #6b7280;
      text-align: center;
    }
    .badge {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
      margin-right: 8px;
    }
    .badge-high { background: #fee2e2; color: #991b1b; }
    .badge-low { background: #dcfce7; color: #166534; }
  </style>
</head>
<body>
  <div class="header">
    <h1>PATHOLOGY REPORT</h1>
    <div class="info">
      <div>
        <strong>RecursiaDx Digital Pathology Lab</strong><br>
        Email: lab@recursiadx.com | Phone: +91 99999 88888
      </div>
      <div style="text-align: right;">
        <strong>Report ID:</strong> ${sample.sampleId || sample._id}<br>
        <strong>Date:</strong> ${new Date().toLocaleDateString()}
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Patient Information</div>
    <div class="field-group">
      <div class="field">
        <div class="field-label">Patient Name</div>
        <div class="field-value">${patientInfo.name || 'N/A'}</div>
      </div>
      <div class="field">
        <div class="field-label">Patient ID</div>
        <div class="field-value">${patientInfo.id || 'N/A'}</div>
      </div>
      <div class="field">
        <div class="field-label">Age</div>
        <div class="field-value">${patientInfo.age || 'N/A'}</div>
      </div>
      <div class="field">
        <div class="field-label">Gender</div>
        <div class="field-value">${patientInfo.gender || 'N/A'}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Sample Information</div>
    <div class="field-group">
      <div class="field">
        <div class="field-label">Sample Type</div>
        <div class="field-value">${sample.sampleType || 'Tissue Biopsy'}</div>
      </div>
      <div class="field">
        <div class="field-label">Collection Date</div>
        <div class="field-value">${sample.sampleInfo?.collectionDate || new Date().toLocaleDateString()}</div>
      </div>
      <div class="field">
        <div class="field-label">Images Analyzed</div>
        <div class="field-value">${sample.images?.length || 0}</div>
      </div>
      <div class="field">
        <div class="field-label">Analysis Status</div>
        <div class="field-value">${sample.status || 'Completed'}</div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">AI Analysis Results</div>
    <div class="alert ${isPositive ? 'alert-positive' : 'alert-negative'}">
      <div class="alert-title">${isPositive ? '⚠️ Abnormal Findings' : '✓ Normal Findings'}</div>
      <p>
        ${isPositive
          ? `AI analysis detected potential abnormalities with ${confidence}% confidence. Tumor probability: ${tumorProb}%. Risk level: ${analysis.risk_level || 'Low'}.`
          : `AI analysis confirms normal findings with ${confidence}% confidence. No significant abnormalities detected.`
        }
      </p>
    </div>
    <div class="field-group">
      <div class="field">
        <div class="field-label">Detection Result</div>
        <div class="field-value">${isPositive ? 'POSITIVE (Abnormal)' : 'NEGATIVE (Normal)'}</div>
      </div>
      <div class="field">
        <div class="field-label">AI Confidence</div>
        <div class="field-value">${confidence}%</div>
      </div>
      <div class="field">
        <div class="field-label">Tumor Probability</div>
        <div class="field-value">${tumorProb}%</div>
      </div>
      <div class="field">
        <div class="field-label">Risk Assessment</div>
        <div class="field-value">
          <span class="badge ${analysis.risk_level?.includes('High') ? 'badge-high' : 'badge-low'}">
            ${analysis.risk_level || 'Low Risk'}
          </span>
        </div>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Clinical Recommendations</div>
    <ul style="padding-left: 20px; margin-top: 10px;">
      ${isPositive ? `
        <li style="margin-bottom: 8px;">Further histopathological examination recommended</li>
        <li style="margin-bottom: 8px;">Consider clinical correlation with patient history</li>
        <li style="margin-bottom: 8px;">Follow-up testing may be indicated</li>
      ` : `
        <li style="margin-bottom: 8px;">No immediate follow-up required based on current findings</li>
        <li style="margin-bottom: 8px;">Continue routine screening as per clinical guidelines</li>
        <li style="margin-bottom: 8px;">Consult with ordering physician for clinical correlation</li>
      `}
    </ul>
  </div>

  <div class="footer">
    <p><strong>RecursiaDx AI-Powered Digital Pathology Platform</strong></p>
    <p>This report was generated using AI-assisted analysis. All findings should be confirmed by a qualified pathologist.</p>
    <p>HIPAA Compliant | Real-time AI Processing | For professional use only</p>
  </div>
</body>
</html>
      `;

      // Launch Puppeteer
      const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
      });

      const page = await browser.newPage();
      await page.setContent(html, { waitUntil: 'networkidle0' });

      // Generate PDF
      const pdf = await page.pdf({
        format: 'A4',
        printBackground: true,
        margin: {
          top: '20px',
          right: '20px',
          bottom: '20px',
          left: '20px'
        }
      });

      await browser.close();

      // Send PDF
      res.contentType('application/pdf');
      res.setHeader('Content-Disposition', `attachment; filename="RecursiaDx_Report_${sample.sampleId || sampleId}.pdf"`);
      res.end(pdf, 'binary');

      console.log('✅ PDF generated successfully');

    } catch (error) {
      console.error('❌ PDF generation error:', error);
      res.status(500).json({
        success: false,
        message: 'Failed to generate PDF',
        error: error.message
      });
    }
  })
);

export default router;