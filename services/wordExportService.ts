export const exportToWord = (htmlContent: string, filename: string, headerHtml: string = '', marginValue: string = '0.6cm') => {
  // 0.6cm in twips (1cm = 567 twips)
  const marginTwips = 340;
  // Line spacing 1.15 (1.15 * 240 twips = 276)
  const lineSpacingTwips = 276;

  const content = `
    <html xmlns:o='urn:schemas-microsoft-com:office:office' 
          xmlns:w='urn:schemas-microsoft-com:office:word' 
          xmlns='http://www.w3.org/TR/REC-html40'>
    <head>
      <meta charset='utf-8'>
      <title>DPSS Export</title>
      <!--[if gte mso 9]>
      <xml>
        <w:WordDocument>
          <w:View>Print</w:View>
          <w:Zoom>100</w:Zoom>
          <w:DoNotOptimizeForBrowser/>
        </w:WordDocument>
      </xml>
      <![endif]-->
      <style>
        @page Section1 {
          size: 8.5in 11.0in;
          margin: ${marginValue} ${marginValue} ${marginValue} ${marginValue};
          mso-header-margin: 0.5in;
          mso-footer-margin: 0.5in;
          mso-paper-source: 0;
        }
        div.Section1 {
          page: Section1;
        }
        body { 
          font-family: 'Times New Roman', serif; 
          font-size: 12pt;
          color: #000;
          margin: 0;
          padding: 0;
        }
        h1, h2, h3, h4 { 
          margin: 0; 
          padding: 0; 
          color: #000; 
          mso-para-margin-top: 0pt; 
          mso-para-margin-bottom: 0pt;
        }
        p, div, li { 
          margin: 0;
          mso-para-margin-top: 0pt;
          mso-para-margin-bottom: 0pt;
          mso-para-margin-left: 0pt;
          mso-para-margin-right: 0pt;
          line-height: 1.15;
          mso-line-height-rule: exactly;
          mso-line-height-alt: 13.8pt; /* 1.15 * 12pt */
        }
        b { font-weight: bold; }
        u { text-decoration: underline; }
        table { 
          border-collapse: collapse; 
          width: 100%; 
          border: 1.5pt solid black; 
          margin: 10pt 0; 
          table-layout: fixed;
          mso-table-lspace: 0pt;
          mso-table-rspace: 0pt;
        }
        td, th { 
          border: 1pt solid black; 
          padding: 6pt; 
          vertical-align: top; 
          overflow: hidden; 
          word-wrap: break-word;
          mso-para-margin-top: 0pt;
          mso-para-margin-bottom: 0pt;
        }
        img { display: block; margin: 0 auto; max-width: 100%; }
        
        /* MS Word Specific Styles */
        p.MsoNormal, li.MsoNormal, div.MsoNormal {
          mso-style-unhide: no;
          mso-style-qformat: yes;
          mso-style-parent: "";
          margin: 0pt;
          margin-bottom: .0001pt;
          mso-pagination: widow-orphan;
          font-size: 12.0pt;
          font-family: "Times New Roman", serif;
          mso-fareast-font-family: "Times New Roman";
          line-height: 115%;
          mso-line-height-rule: exactly;
        }

        /* Force Section Setup */
        @xml {
          <w:WordDocument>
            <w:View>Print</w:View>
            <w:Zoom>100</w:Zoom>
          </w:WordDocument>
          <w:PageSetup>
            <w:Margins w:left="${marginTwips}" w:right="${marginTwips}" w:top="${marginTwips}" w:bottom="${marginTwips}" w:header="720" w:footer="720" w:gutter="0"/>
            <w:LineSpacing w:line="${lineSpacingTwips}" w:lineRule="auto"/>
          </w:PageSetup>
        }
      </style>
    </head>
    <body>
      <div class="Section1">
        ${headerHtml}
        <div style="margin-top: 10pt;">
          ${htmlContent}
        </div>
      </div>
    </body>
    </html>`;

  const blob = new Blob(['\ufeff', content], {
    type: 'application/msword'
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.doc`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};