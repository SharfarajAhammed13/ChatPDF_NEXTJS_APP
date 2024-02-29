import { Pinecone } from '@pinecone-database/pinecone';

import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import md5 from 'md5';
import { downloadFromS3 } from './s3-server';
import {Document, RecursiveCharacterTextSplitter} from '@pinecone-database/doc-splitter'
import { getEmbeddings } from './embeddings';
import { Vector } from '@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch';
import { convertToAscii } from './utils';


let pinecone: Pinecone | null = null;

export const getPineconeClient = async () => {
    if (!pinecone) {
        pinecone = new Pinecone();
        await pinecone.init({
            environment: process.env.PINECONE_ENVIRONMENT,
            apiKey: process.env.PINECONE_API_KEY,
        });
    }
    return pinecone
}

type PDFPage = {
    pageContent:string;
    metadata: {
        loc: {pageNumber: number;}
    }
}

class PineconeUtils {
    static async chunkedUpsert(
      index: any,
      vectors: Vector[],
      namespace: string,
      batchSize: number
    ) {
      const chunks = [];
      for (let i = 0; i < vectors.length; i += batchSize) {
        const chunk = vectors.slice(i, i + batchSize);
        chunks.push(chunk);
      }
  
      for (const chunk of chunks) {
        await index.upsert(chunk, namespace);
      }
    }
  }
  

export async function loadS3IntoPinecone(fileKey:string) {
    //1. Obtain the pdf - download and read from pdf
    console.log('Downloading s into file system')
    const file_name = await downloadFromS3(fileKey);
    if (!file_name) {
        throw new Error("Could not download from S3");
    }
    const loader = new PDFLoader(file_name);
    const pages = await loader.load() as PDFPage[];
    return pages;

    // 2. Split and segment the pdf into pages
    // pages = array(13)
    const documents = await Promise.all(pages.map(prepareDocument));

    // 3 vectorise and embed individual document
    const vectors = await Promise.all(documents.flat().map(embedDocument))

    //4 upload to pinecone
    const client = await getPineconeClient()
    const pineconeIndex = client.Index('chatpdf')

    console.log('Inserting vector into pinecode')
    const namespace = convertToAscii(fileKey)
    PineconeUtils.chunkedUpsert(pineconeIndex, vectors, namespace, 10)

}

async function embedDocument(doc: Document) {
    try {
        const embeddings = await getEmbeddings(doc.pageContent)
        const hash = md5(doc.pageContent)
        return {
            id: hash,
            values: embeddings,
            metadata: {
                text: doc.metadata.text,
                pageNumber: doc.metadata.pageNumber
            }
        }  as Vector
    } catch (error) {
        console.log('Error embedding document', error)
        throw error
    }
}

export const truncateStringByBytes = (str: string, bytes: number) => {
    const enc = new TextEncoder()
    return new TextDecoder('utf-8').decode(enc.encode(str).slice(0,bytes))
}

async function prepareDocument(page: PDFPage) {
    let { pageContent, metadata } = page;
    pageContent = pageContent.replace(/\n/g, '');
    //split the docs 
    const splitter = new RecursiveCharacterTextSplitter()
    const docs = await splitter.splitDocuments([
        new Document({
            pageContent,
            metadata: {
                pageNumber: metadata.loc.pageNumber,
                text: truncateStringByBytes(pageContent, 36000)
            }
        })
    ])
    return docs
}
