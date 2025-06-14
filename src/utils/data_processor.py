from typing import Dict, Any, List
import json
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, XSD

class VCDataProcessor:
    @staticmethod
    def json_to_text(vc_json: Dict[str, Any]) -> str:
        """Convert a Verifiable Credential JSON to a text representation."""
        text_parts = []
        
        # Add type information
        if "type" in vc_json:
            if isinstance(vc_json["type"], list):
                text_parts.append(f"Types: {', '.join(vc_json['type'])}")
            else:
                text_parts.append(f"Type: {vc_json['type']}")
        
        # Add issuer information
        if "issuer" in vc_json:
            text_parts.append(f"Issuer: {vc_json['issuer']}")
        
        # Add issuance date
        if "issuanceDate" in vc_json:
            text_parts.append(f"Issued on: {vc_json['issuanceDate']}")
        
        # Add expiration date if present
        if "expirationDate" in vc_json:
            text_parts.append(f"Expires on: {vc_json['expirationDate']}")
        
        # Add credential subject information
        if "credentialSubject" in vc_json:
            subject = vc_json["credentialSubject"]
            text_parts.append("Credential Subject:")
            
            # Handle both single subject and array of subjects
            subjects = [subject] if isinstance(subject, dict) else subject
            
            for subj in subjects:
                for key, value in subj.items():
                    if key != "id":  # Skip the ID field
                        text_parts.append(f"  {key}: {value}")
        
        return "\n".join(text_parts)

    @staticmethod
    def rdf_to_text(rdf_data: str) -> str:
        """Convert RDF data to a text representation."""
        g = Graph()
        g.parse(data=rdf_data, format="turtle")
        
        text_parts = []
        
        # Extract type information
        for s, p, o in g.triples((None, RDF.type, None)):
            text_parts.append(f"Type: {o}")
        
        # Extract other properties
        for s, p, o in g.triples((None, None, None)):
            if p != RDF.type:  # Skip type triples as they're already handled
                if isinstance(o, Literal):
                    text_parts.append(f"{p}: {o}")
                else:
                    text_parts.append(f"{p}: {o}")
        
        return "\n".join(text_parts)

    @staticmethod
    def _convert_metadata_value(value: Any) -> str:
        """Convert metadata values to strings, handling lists and other complex types."""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)

    @staticmethod
    def process_vc_data(vc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Verifiable Credential data for vector storage."""
        # Extract text representation
        if "json" in vc_data:
            text_content = VCDataProcessor.json_to_text(vc_data["json"])
        elif "rdf" in vc_data:
            text_content = VCDataProcessor.rdf_to_text(vc_data["rdf"])
        else:
            raise ValueError("VC data must contain either 'json' or 'rdf' field")
        
        # Create metadata with string values
        metadata = {
            "id": vc_data.get("id", ""),
            "type": VCDataProcessor._convert_metadata_value(vc_data.get("type", [])),
            "issuer": vc_data.get("issuer", ""),
            "issuance_date": vc_data.get("issuanceDate", ""),
            "expiration_date": vc_data.get("expirationDate", "")
        }
        
        return {
            "text": text_content,
            "metadata": metadata
        } 