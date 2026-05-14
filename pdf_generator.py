import io
from fpdf import FPDF, HTMLMixin

class PDF(FPDF, HTMLMixin):
    pass

def generate_pdf_from_html(html_content: str) -> bytes:
    try:
        # FPDF HTML parser doesn't support complex CSS, so we do a very basic clean up.
        # We will strip out the base64 images to prevent crashes and just provide the text report.
        import re
        # Remove <style> blocks
        clean_html = re.sub(r'<style.*?>.*?</style>', '', html_content, flags=re.DOTALL)
        # Remove base64 image tags as fpdf2 HTMLParser struggles with them
        clean_html = re.sub(r'<img src="data:image.*?/>', '[IMAGE OMITTED IN PDF]', clean_html)
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("helvetica", size=12)
        pdf.write_html(clean_html)
        
        return pdf.output(dest='S')
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return b""
